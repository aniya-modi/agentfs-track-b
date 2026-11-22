"""Run selector proposals locally and return compact evidence to the agent graph.

The expression matrix and labels are captured by this local runner. Only
selected feature identifiers and fold-validated summary metrics are returned to
LangGraph; the Ollama role prompts never receive sample-level measurements.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.impute import SimpleImputer

from .evaluation import run_repeated_nested_cv
from .schemas import FeatureSelectionConfig, SelectionCandidateResult
from .tool_registry import FeatureSelectionToolRegistry


def _param_grid(config: FeatureSelectionConfig, n_features: int) -> dict[str, list[Any]]:
    """Small, method-specific inner-CV grids for the demo workflow."""
    common: dict[str, list[Any]] = {
        "selector__top_k": [min(config.k, n_features)],
        "classifier__C": [0.1, 1.0, 10.0],
    }
    if config.method == "mrmr":
        pools = sorted({min(n_features, max(config.k, 250)), min(n_features, 500)})
        common["selector__candidate_pool"] = pools
    elif config.method == "lasso":
        common["selector__C"] = [0.5, 1.0]
    elif config.method == "hypergraph_centrality":
        common["selector__spectral_weight"] = [0.2, 0.4, 0.6]
    return common


def _pathway_summary(
    incidence: Any | None, selected_indices: Sequence[int]
) -> tuple[int, int]:
    if incidence is None:
        return 0, 0
    matrix = sparse.csr_matrix(incidence)
    if not selected_indices:
        return 0, 0
    selected = matrix[np.asarray(selected_indices, dtype=int)]
    return int(np.count_nonzero(np.asarray(selected.sum(axis=0)).ravel())), int(
        np.count_nonzero(np.asarray(selected.sum(axis=1)).ravel())
    )


def _mean_pairwise_correlation(X: np.ndarray, indices: Sequence[int]) -> float | None:
    if len(indices) < 2:
        return None
    selected = X[:, np.asarray(indices, dtype=int)]
    correlation = np.corrcoef(selected, rowvar=False)
    if np.ndim(correlation) < 2:
        return None
    upper = correlation[np.triu_indices_from(correlation, k=1)]
    finite = np.abs(upper[np.isfinite(upper)])
    return float(np.clip(finite.mean(), 0.0, 1.0)) if finite.size else None


def make_local_selection_executor(
    expression: pd.DataFrame,
    labels: pd.Series,
    *,
    registry: FeatureSelectionToolRegistry,
    n_splits: int = 3,
    n_repeats: int = 1,
    inner_splits: int = 2,
    random_state: int = 42,
    max_candidates: int = 5,
):
    """Create a closure that evaluates proposals and selects features locally.

    Evaluation uses nested CV with all learned preprocessing and feature
    selection inside each fold. After that comparison, each proposal's final
    feature list is fitted on the full supplied dataset for exploratory output.
    Those full-data lists are not used to calculate the CV metrics.
    """
    if not isinstance(expression, pd.DataFrame) or expression.empty:
        raise ValueError("expression must be a non-empty pandas DataFrame.")
    if not expression.columns.is_unique:
        raise ValueError("Feature identifiers must be unique.")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in expression.dtypes):
        raise TypeError("All selector inputs must be numeric columns.")
    if not isinstance(labels, pd.Series) or len(labels) != len(expression):
        raise ValueError("labels must be a Series with one value per expression row.")
    if labels.isna().any():
        raise ValueError("Labels cannot contain missing values for supervised selection.")
    if labels.nunique(dropna=True) < 2:
        raise ValueError("At least two target classes are required for supervised selection.")
    if min(labels.value_counts()) < n_splits:
        raise ValueError("Each class must contain at least n_splits samples for outer CV.")
    if min(labels.value_counts()) < inner_splits + 1:
        raise ValueError("Each class needs enough samples for the nested inner CV folds.")
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive.")

    X = expression.to_numpy(dtype=float, na_value=np.nan)
    X[~np.isfinite(X)] = np.nan
    all_missing = np.all(np.isnan(X), axis=0)
    if all_missing.any():
        empty_ids = [str(expression.columns[index]) for index in np.flatnonzero(all_missing)]
        raise ValueError(
            "Remove features with no finite observations before selection: "
            f"{empty_ids[:10]}"
        )
    y = labels.to_numpy()
    feature_ids = [str(item) for item in expression.columns]

    def execute(configs: list[FeatureSelectionConfig]) -> list[SelectionCandidateResult]:
        results: list[SelectionCandidateResult] = []
        seen: set[tuple[str, int]] = set()
        available = {tool["name"] for tool in registry.list_tools() if tool["available"]}
        for original in configs:
            config = FeatureSelectionConfig.model_validate(original).model_copy(
                update={"k": min(original.k, X.shape[1]), "two_stage_cascade": False}
            )
            key = (config.method, config.k)
            if key in seen:
                continue
            seen.add(key)
            if config.method not in available:
                continue

            selector_factory = lambda cfg=config: registry.build_selector(cfg)
            cv = run_repeated_nested_cv(
                X,
                y,
                selector_factory=selector_factory,
                param_grid=_param_grid(config, X.shape[1]),
                n_splits=n_splits,
                n_repeats=n_repeats,
                inner_splits=inner_splits,
                random_state=random_state,
            )

            imputed_X = SimpleImputer(strategy="median").fit_transform(X)
            selected = registry.execute(
                config,
                imputed_X,
                y,
                feature_ids=feature_ids,
            )
            pathway_count, pathway_feature_count = _pathway_summary(
                registry.H, selected.selected_indices
            )
            results.append(
                SelectionCandidateResult(
                    config=config,
                    selected_feature_ids=selected.selected_feature_ids,
                    cv_accuracy=cv["mean_accuracy"],
                    cv_balanced_accuracy=cv["mean_balanced_accuracy"],
                    cv_macro_f1=cv["mean_macro_f1"],
                    selection_stability=cv["jaccard_stability"],
                    pathway_count=pathway_count,
                    pathway_supported_feature_count=pathway_feature_count,
                    mean_pairwise_correlation=_mean_pairwise_correlation(
                        imputed_X, selected.selected_indices
                    ),
                )
            )
            if len(results) >= max_candidates:
                break
        if not results:
            raise ValueError("No proposed selector was executable for this dataset.")
        return results

    return execute
