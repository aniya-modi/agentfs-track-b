"""Fold-local mRMR and L1 logistic-regression feature selectors."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class MRMRSelector(BaseEstimator, TransformerMixin):
    """Greedily select relevant features while penalizing repeated signals.

    The score is normalized ANOVA relevance minus the mean absolute
    correlation with features already selected. All learned quantities are
    estimated during ``fit`` and therefore can be kept inside CV folds.
    """

    def __init__(self, top_k: int = 50, candidate_pool: int = 500):
        self.top_k = top_k
        self.candidate_pool = candidate_pool

    def fit(self, X: Any, y: Any) -> MRMRSelector:
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)
        if X_arr.ndim != 2:
            raise ValueError("X must be a two-dimensional samples-by-features matrix.")
        if len(y_arr) != X_arr.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")
        if X_arr.shape[1] == 0:
            raise ValueError("X must contain at least one feature.")
        if int(self.top_k) < 1 or int(self.candidate_pool) < 1:
            raise ValueError("top_k and candidate_pool must both be positive.")

        n_features = X_arr.shape[1]
        k = min(int(self.top_k), n_features)
        pool_size = min(n_features, max(k, int(self.candidate_pool)))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f_scores, _ = f_classif(X_arr, y_arr)
        f_scores = np.nan_to_num(f_scores, nan=0.0, posinf=1e12, neginf=0.0)
        maximum = float(np.max(f_scores))
        relevance = f_scores / maximum if maximum > 0 else np.zeros_like(f_scores)

        candidates = np.argsort(-relevance, kind="mergesort")[:pool_size]
        correlations = np.corrcoef(X_arr[:, candidates], rowvar=False)
        if np.ndim(correlations) == 0:
            correlations = np.array([[1.0]])
        absolute_correlations = np.nan_to_num(np.abs(correlations), nan=0.0)

        selected_local = [0]
        remaining = list(range(1, pool_size))
        while remaining and len(selected_local) < k:
            redundancy = np.mean(
                absolute_correlations[np.ix_(remaining, selected_local)], axis=1
            )
            scores = relevance[candidates[remaining]] - redundancy
            best_position = int(np.argmax(scores))
            selected_local.append(remaining.pop(best_position))

        self.support_ = np.sort(candidates[np.asarray(selected_local, dtype=int)])
        self.n_features_in_ = n_features
        return self

    def transform(self, X: Any) -> np.ndarray:
        X_arr = np.asarray(X)
        if X_arr.ndim != 2 or X_arr.shape[1] != self.n_features_in_:
            raise ValueError("X must have the same number of features seen during fit.")
        return X_arr[:, self.support_]

    def get_support(self, indices: bool = False) -> np.ndarray:
        if indices:
            return self.support_.copy()
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[self.support_] = True
        return mask


class LassoSelector(BaseEstimator, TransformerMixin):
    """Rank features by absolute coefficients from L1 logistic regression."""

    def __init__(self, top_k: int = 50, C: float = 1.0, random_state: int = 42):
        self.top_k = top_k
        self.C = C
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> LassoSelector:
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)
        if X_arr.ndim != 2:
            raise ValueError("X must be a two-dimensional samples-by-features matrix.")
        if len(y_arr) != X_arr.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")
        if X_arr.shape[1] == 0:
            raise ValueError("X must contain at least one feature.")
        if int(self.top_k) < 1:
            raise ValueError("top_k must be positive.")
        if float(self.C) <= 0:
            raise ValueError("C must be positive.")

        n_features = X_arr.shape[1]
        k = min(int(self.top_k), n_features)
        self.scaler_ = StandardScaler().fit(X_arr)
        Z = self.scaler_.transform(X_arr)

        self.classifier_ = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=float(self.C),
            class_weight="balanced",
            random_state=self.random_state,
            max_iter=1000,
        ).fit(Z, y_arr)
        coefficients = np.max(np.abs(self.classifier_.coef_), axis=0)
        self.support_ = np.sort(np.argsort(-coefficients, kind="mergesort")[:k])
        self.n_features_in_ = n_features
        return self

    def transform(self, X: Any) -> np.ndarray:
        X_arr = np.asarray(X)
        if X_arr.ndim != 2 or X_arr.shape[1] != self.n_features_in_:
            raise ValueError("X must have the same number of features seen during fit.")
        return X_arr[:, self.support_]

    def get_support(self, indices: bool = False) -> np.ndarray:
        if indices:
            return self.support_.copy()
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[self.support_] = True
        return mask
