"""Spectral hypergraph node centrality feature selector for microarray data."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif


class HypergraphCentralitySelector(BaseEstimator, TransformerMixin):
    """Rank probes via fold-local class relevance + spectral hypergraph centrality.

    Computes the normalized hypergraph random-walk operator:
        S = D_v^{-1/2} H D_e^{-1} H^T D_v^{-1/2}
    and extracts the dominant spectral centrality vector combined with fold-local
    ANOVA F-scores (so every fit remains strictly inside training folds).
    """

    def __init__(
        self,
        H: Any,
        top_k: int = 50,
        spectral_weight: float = 0.4,
        damping: float = 0.85,
        n_iter: int = 30,
    ):
        self.H = H
        self.top_k = top_k
        self.spectral_weight = spectral_weight
        self.damping = damping
        self.n_iter = n_iter

    def fit(self, X: Any, y: Any) -> HypergraphCentralitySelector:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n_features = X.shape[1]

        H_csr = sparse.csr_matrix(self.H, dtype=float)
        if H_csr.shape[0] != n_features:
            raise ValueError(
                f"Incidence matrix rows ({H_csr.shape[0]}) != input features ({n_features})"
            )

        # 1. Fold-local supervised gene relevance (normalized ANOVA F-score)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f_scores, _ = f_classif(X, y)
        f_scores = np.nan_to_num(f_scores, nan=0.0, posinf=1e12, neginf=0.0)
        max_f = float(np.max(f_scores))
        stat_score = f_scores / max_f if max_f > 0 else np.zeros(n_features, dtype=float)

        # 2. Sparse normalized hypergraph operator S = D_v^{-1/2} H D_e^{-1} H^T D_v^{-1/2}
        d_v = np.asarray(H_csr.sum(axis=1)).ravel()
        d_e = np.asarray(H_csr.sum(axis=0)).ravel()

        inv_sqrt_dv = np.zeros_like(d_v)
        mask_v = d_v > 0
        inv_sqrt_dv[mask_v] = 1.0 / np.sqrt(d_v[mask_v])

        inv_de = np.zeros_like(d_e)
        mask_e = d_e > 0
        inv_de[mask_e] = 1.0 / d_e[mask_e]

        D_v_inv_sqrt = sparse.diags(inv_sqrt_dv)
        D_e_inv = sparse.diags(inv_de)
        P_left = D_v_inv_sqrt @ H_csr @ D_e_inv
        P_right = H_csr.T @ D_v_inv_sqrt

        # 3. Personalized spectral centrality seeded by fold-local statistical signal
        #    r^{(t+1)} = alpha * (P_left @ (P_right @ r^{(t)})) + (1 - alpha) * stat_score
        alpha = float(self.damping)
        centrality = stat_score.copy()
        for _ in range(int(self.n_iter)):
            propagated = P_left @ (P_right @ centrality)
            centrality = alpha * propagated + (1.0 - alpha) * stat_score

        max_c = float(np.max(centrality))
        norm_centrality = centrality / max_c if max_c > 0 else np.zeros(n_features, dtype=float)

        w = float(self.spectral_weight)
        combined = (1.0 - w) * stat_score + w * norm_centrality

        k = min(int(self.top_k), n_features)
        self.support_ = np.sort(np.argsort(-combined, kind="mergesort")[:k])
        self.scores_ = combined
        self.n_features_in_ = n_features
        return self

    def transform(self, X: Any) -> np.ndarray:
        return np.asarray(X)[:, self.support_]

    def get_support(self, indices: bool = False) -> np.ndarray:
        if indices:
            return self.support_.copy()
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[self.support_] = True
        return mask
