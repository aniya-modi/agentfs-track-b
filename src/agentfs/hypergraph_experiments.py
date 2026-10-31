"""Fold-local biological/statistical selectors for microarray experiments.

These prototypes are designed for sklearn ``Pipeline`` and ``GridSearchCV`` so
that imputation, gene scoring, pathway aggregation, and correlation estimates
are learned from training folds only.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def _rank_scores(X: np.ndarray, y: Any, incidence: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized gene F-score ranks and pathway-propagated ranks."""
    if incidence.shape[0] != X.shape[1]:
        raise ValueError(
            f"Incidence matrix has {incidence.shape[0]} rows for {X.shape[1]} input features"
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f_scores, _ = f_classif(X, y)
    f_scores = np.nan_to_num(f_scores, nan=0.0, posinf=1e12, neginf=0.0)
    gene_score = rankdata(f_scores, method="average")
    gene_score = (gene_score - 1.0) / max(len(gene_score) - 1, 1)

    pathway_size = np.asarray(incidence.sum(axis=0)).ravel()
    pathway_score = np.asarray(incidence.T @ gene_score).ravel()
    pathway_score /= np.maximum(pathway_size, 1)
    degree = np.asarray(incidence.sum(axis=1)).ravel()
    propagated = np.asarray(incidence @ pathway_score).ravel()
    propagated /= np.maximum(degree, 1)
    propagated[degree == 0] = gene_score[degree == 0]
    return gene_score, propagated


def _correlation_matrix(X: np.ndarray) -> np.ndarray:
    """Absolute correlation for a small candidate set, with constant columns safe."""
    corr = np.corrcoef(X, rowvar=False)
    if np.ndim(corr) == 0:
        corr = np.array([[1.0]])
    return np.nan_to_num(np.abs(corr), nan=0.0, posinf=0.0, neginf=0.0)


class PathwayMeanActivity(BaseEstimator, TransformerMixin):
    """Aggregate fold-standardized probe expression into pathway activities.

    Place a training-fold ``StandardScaler`` before this transformer in a
    pipeline. Each output feature is the mean standardized expression of the
    probes belonging to one pathway.
    """

    def __init__(self, H: Any):
        self.H = H

    def fit(self, X: Any, y: Any = None) -> PathwayMeanActivity:
        if self.H.shape[0] != X.shape[1]:
            raise ValueError("Incidence matrix rows must match expression features")
        self.pathway_sizes_ = np.asarray(self.H.sum(axis=0)).ravel()
        return self

    def transform(self, X: Any) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        activity = np.asarray(self.H.T @ X.T).T
        return activity / np.maximum(self.pathway_sizes_, 1)[None, :]


class BioStatRedundancySelector(BaseEstimator, TransformerMixin):
    """Blend statistical and pathway ranks, then hard-prune correlated probes."""

    def __init__(
        self,
        H: Any,
        top_k: int = 50,
        alpha: float = 0.5,
        corr_threshold: float = 0.9,
        candidate_multiplier: int = 3,
    ):
        self.H = H
        self.top_k = top_k
        self.alpha = alpha
        self.corr_threshold = corr_threshold
        self.candidate_multiplier = candidate_multiplier

    def fit(self, X: Any, y: Any) -> BioStatRedundancySelector:
        X = np.asarray(X, dtype=float)
        gene_score, pathway_score = _rank_scores(X, y, self.H)
        relevance = self.alpha * gene_score + (1.0 - self.alpha) * pathway_score
        k = min(int(self.top_k), X.shape[1])
        n_candidates = min(X.shape[1], max(k, k * int(self.candidate_multiplier)))
        candidates = np.argsort(-relevance, kind="mergesort")[:n_candidates]
        corr = _correlation_matrix(X[:, candidates])

        chosen: list[int] = []
        for local_index in range(len(candidates)):
            if all(corr[local_index, prior] < self.corr_threshold for prior in chosen):
                chosen.append(local_index)
                if len(chosen) == k:
                    break
        self.support_ = np.sort(candidates[np.asarray(chosen, dtype=int)])
        self.n_features_in_ = X.shape[1]
        self.mean_abs_corr_selected_ = _mean_pairwise_corr(X[:, self.support_])
        return self

    def transform(self, X: Any) -> np.ndarray:
        return np.asarray(X)[:, self.support_]

    def get_support(self, indices: bool = False) -> np.ndarray:
        if indices:
            return self.support_.copy()
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[self.support_] = True
        return mask


class BioStatSoftRedundancySelector(BaseEstimator, TransformerMixin):
    """Greedily trade off combined relevance and maximum selected-gene correlation."""

    def __init__(
        self,
        H: Any,
        top_k: int = 50,
        alpha: float = 0.5,
        gamma: float = 0.25,
        candidate_multiplier: int = 3,
    ):
        self.H = H
        self.top_k = top_k
        self.alpha = alpha
        self.gamma = gamma
        self.candidate_multiplier = candidate_multiplier

    def fit(self, X: Any, y: Any) -> BioStatSoftRedundancySelector:
        X = np.asarray(X, dtype=float)
        gene_score, pathway_score = _rank_scores(X, y, self.H)
        relevance = self.alpha * gene_score + (1.0 - self.alpha) * pathway_score
        k = min(int(self.top_k), X.shape[1])
        n_candidates = min(X.shape[1], max(k, k * int(self.candidate_multiplier)))
        candidates = np.argsort(-relevance, kind="mergesort")[:n_candidates]
        corr = _correlation_matrix(X[:, candidates])

        remaining = list(range(len(candidates)))
        chosen: list[int] = []
        while remaining and len(chosen) < k:
            if not chosen or self.gamma == 0:
                adjusted = relevance[candidates[remaining]]
            else:
                redundancy = np.max(corr[np.ix_(remaining, chosen)], axis=1)
                adjusted = relevance[candidates[remaining]] - self.gamma * redundancy
            best_position = int(np.argmax(adjusted))
            chosen.append(remaining.pop(best_position))

        self.support_ = np.sort(candidates[np.asarray(chosen, dtype=int)])
        self.n_features_in_ = X.shape[1]
        self.mean_abs_corr_selected_ = _mean_pairwise_corr(X[:, self.support_])
        return self

    def transform(self, X: Any) -> np.ndarray:
        return np.asarray(X)[:, self.support_]

    def get_support(self, indices: bool = False) -> np.ndarray:
        if indices:
            return self.support_.copy()
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[self.support_] = True
        return mask


def _mean_pairwise_corr(X: np.ndarray) -> float:
    if X.shape[1] < 2:
        return 0.0
    corr = np.corrcoef(X, rowvar=False)
    upper = corr[np.triu_indices_from(corr, k=1)]
    return float(np.nanmean(np.abs(upper)))


class HypergraphLapLogistic(ClassifierMixin, BaseEstimator):
    """Binary logistic model with fold-local gene screening and Laplacian penalty.

    The objective is weighted logistic loss plus ``||beta||²/(2C)`` and
    ``lambda_h * beta.T @ L_H @ beta / 2``. The hypergraph Laplacian is built
    only over the selected probes for each fit.
    """

    def __init__(
        self,
        H: Any,
        top_k: int = 50,
        C: float = 1.0,
        lambda_h: float = 0.1,
        max_iter: int = 500,
    ):
        self.H = H
        self.top_k = top_k
        self.C = C
        self.lambda_h = lambda_h
        self.max_iter = max_iter

    def fit(self, X: Any, y: Any) -> HypergraphLapLogistic:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("HypergraphLapLogistic supports binary targets only")

        self.imputer_ = SimpleImputer(strategy="median").fit(X)
        X_imputed = self.imputer_.transform(X)
        if self.H.shape[0] != X_imputed.shape[1]:
            raise ValueError("Incidence matrix rows must match expression features")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f_scores, _ = f_classif(X_imputed, y)
        f_scores = np.nan_to_num(f_scores, nan=0.0, posinf=1e12, neginf=0.0)
        k = min(int(self.top_k), X_imputed.shape[1])
        self.support_ = np.sort(np.argsort(f_scores, kind="mergesort")[-k:])

        X_selected = X_imputed[:, self.support_]
        self.scaler_ = StandardScaler().fit(X_selected)
        Z = self.scaler_.transform(X_selected)
        self.laplacian_ = _selected_hypergraph_laplacian(self.H[self.support_, :])

        y_binary = (y == self.classes_[1]).astype(float)
        counts = np.bincount(y_binary.astype(int), minlength=2).astype(float)
        sample_weight = len(y_binary) / (2.0 * counts[y_binary.astype(int)])
        n_samples, n_features = Z.shape
        lambda_l2 = 1.0 / float(self.C)
        lambda_h = float(self.lambda_h)
        laplacian = self.laplacian_

        def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
            intercept = parameters[0]
            beta = parameters[1:]
            logits = intercept + Z @ beta
            probabilities = expit(logits)
            data_loss = np.mean(
                sample_weight * (np.logaddexp(0.0, logits) - y_binary * logits)
            )
            l2_loss = 0.5 * lambda_l2 * np.dot(beta, beta)
            hypergraph_loss = 0.5 * lambda_h * beta @ laplacian @ beta
            residual = sample_weight * (probabilities - y_binary) / n_samples
            gradient = np.empty(n_features + 1)
            gradient[0] = residual.sum()
            gradient[1:] = (
                Z.T @ residual + lambda_l2 * beta + lambda_h * (laplacian @ beta)
            )
            return data_loss + l2_loss + hypergraph_loss, gradient

        result = minimize(
            objective,
            np.zeros(n_features + 1),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": int(self.max_iter), "ftol": 1e-8},
        )
        self.intercept_ = float(result.x[0])
        self.coef_ = result.x[1:].reshape(1, -1)
        self.n_iter_ = np.array([result.nit])
        self.optimization_success_ = bool(result.success)
        self.n_features_in_ = X.shape[1]
        return self

    def decision_function(self, X: Any) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        X_imputed = self.imputer_.transform(X)
        Z = self.scaler_.transform(X_imputed[:, self.support_])
        return self.intercept_ + Z @ self.coef_[0]

    def predict(self, X: Any) -> np.ndarray:
        positive = self.decision_function(X) >= 0
        return self.classes_[positive.astype(int)]


def _selected_hypergraph_laplacian(incidence: Any) -> np.ndarray:
    """Construct the normalized vertex Laplacian for selected probes."""
    H = incidence.toarray().astype(float)
    edge_sizes = H.sum(axis=0)
    active_edges = edge_sizes > 0
    H = H[:, active_edges]
    edge_sizes = edge_sizes[active_edges]
    degree = H.sum(axis=1)
    covered = degree > 0
    inv_sqrt_degree = np.zeros_like(degree)
    inv_sqrt_degree[covered] = 1.0 / np.sqrt(degree[covered])
    if H.shape[1] == 0:
        return np.zeros((H.shape[0], H.shape[0]), dtype=float)
    normalized_H = H * inv_sqrt_degree[:, None]
    similarity = (normalized_H / edge_sizes[None, :]) @ normalized_H.T
    return np.diag(covered.astype(float)) - similarity
