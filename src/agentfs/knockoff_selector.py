"""Exploratory supervised screening followed by a knockoff selection step."""

from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler

try:
    from knockpy import KnockoffFilter
except ImportError as exc:  # pragma: no cover - depends on optional extra
    KnockoffFilter = None
    _KNOCKPY_IMPORT_ERROR = exc


class TwoStageKnockoffFilter(BaseEstimator, TransformerMixin):
    """Screen by ANOVA/MI and pass candidates to knockpy's Gaussian filter.

    The first stage uses the labels, so this implementation does not claim
    formal Model-X FDR control. ``fdr`` is the threshold supplied to knockpy;
    it is reported as a requested threshold only. No features are forced into
    the result when the knockoff stage returns an empty set.
    """

    def __init__(
        self,
        fdr: float = 0.10,
        stage1_k: int = 200,
        screening_method: Literal["anova", "mi"] = "anova",
        random_state: int = 42,
    ):
        self.fdr = fdr
        self.stage1_k = stage1_k
        self.screening_method = screening_method
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> TwoStageKnockoffFilter:
        if KnockoffFilter is None:
            raise ImportError(
                "knockpy is required for TwoStageKnockoffFilter. "
                "Install the project analysis extra with `pip install -e '.[analysis]'`."
            ) from _KNOCKPY_IMPORT_ERROR

        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)
        if X_arr.ndim != 2:
            raise ValueError("X must be a two-dimensional samples-by-features matrix.")
        if y_arr.ndim != 1 or len(y_arr) != X_arr.shape[0]:
            raise ValueError("y must be one-dimensional and match the rows of X.")
        if X_arr.shape[1] == 0:
            raise ValueError("X must contain at least one feature.")
        if not 0.0 < float(self.fdr) < 1.0:
            raise ValueError("fdr must be strictly between 0 and 1.")
        if int(self.stage1_k) < 1:
            raise ValueError("stage1_k must be positive.")
        if self.screening_method not in {"anova", "mi"}:
            raise ValueError("screening_method must be 'anova' or 'mi'.")

        n_features = X_arr.shape[1]
        stage1_k = min(int(self.stage1_k), n_features)
        if self.screening_method == "mi":
            score_function = lambda features, labels: mutual_info_classif(
                features, labels, random_state=self.random_state
            )
        else:
            score_function = f_classif

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            screener = SelectKBest(score_func=score_function, k=stage1_k).fit(X_arr, y_arr)
        screened_indices = screener.get_support(indices=True)
        X_screened = screener.transform(X_arr)
        Z = StandardScaler().fit_transform(X_screened)

        knockoff_filter = KnockoffFilter(fstat="lasso", ksampler="gaussian")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rejections = knockoff_filter.forward(X=Z, y=y_arr, fdr=float(self.fdr))

        selected_local = np.flatnonzero(np.asarray(rejections, dtype=bool))
        self.support_ = np.sort(screened_indices[selected_local])
        self.stage1_support_ = screened_indices
        self.knockoff_w_ = getattr(knockoff_filter, "W", None)
        self.n_features_in_ = n_features
        self.n_features_screened_ = len(screened_indices)
        self.n_features_selected_ = len(self.support_)
        self.fdr_target_ = float(self.fdr)
        self.formal_fdr_control_ = False
        self.screening_method_ = self.screening_method
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
