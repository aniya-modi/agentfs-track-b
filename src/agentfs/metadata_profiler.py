"""Create JSON-safe structural summaries of gene-expression matrices.

The profiler returns aggregate statistics only. It never serializes expression
values or sample identifiers. If its statistics will influence evaluation
settings, run it on the relevant training fold rather than on the full dataset.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _correlation_summary(
    expression: pd.DataFrame,
    *,
    max_features: int,
) -> dict[str, Any]:
    """Summarize absolute pairwise correlations on a deterministic feature subset."""
    if max_features < 2:
        raise ValueError("max_correlation_features must be at least 2.")

    clean_expression = expression.replace([np.inf, -np.inf], np.nan)
    variances = clean_expression.var(axis=0, skipna=True).replace([np.inf, -np.inf], np.nan)
    usable = variances.dropna().sort_values(ascending=False, kind="stable")
    selected = list(usable.index[:max_features])
    if len(selected) < 2:
        return {
            "features_used": len(selected),
            "feature_limit": max_features,
            "mean_absolute_pairwise_correlation": None,
            "max_absolute_pairwise_correlation": None,
        }

    correlations = clean_expression.loc[:, selected].corr().to_numpy(dtype=float)
    upper = correlations[np.triu_indices_from(correlations, k=1)]
    finite = np.abs(upper[np.isfinite(upper)])
    return {
        "features_used": len(selected),
        "feature_limit": max_features,
        "mean_absolute_pairwise_correlation": (
            float(finite.mean()) if finite.size else None
        ),
        "max_absolute_pairwise_correlation": (
            float(finite.max()) if finite.size else None
        ),
    }


def profile_dataset(
    expression: pd.DataFrame,
    labels: pd.Series,
    *,
    variance_threshold: float = 1e-8,
    max_correlation_features: int = 256,
    dataset_id: str | None = None,
) -> dict[str, Any]:
    """Return dataset geometry, quality indicators, label counts, and correlation.

    ``expression`` must have one sample per row and one numeric feature per
    column. ``labels`` must have one value per row. The returned dictionary can
    be serialized with :func:`json.dumps` and contains no sample-level values.

    Correlation is summarized over the highest-variance features, capped by
    ``max_correlation_features`` to keep profiling practical for p >> n data.
    """
    if not isinstance(expression, pd.DataFrame):
        raise TypeError("expression must be a pandas DataFrame.")
    if not isinstance(labels, pd.Series):
        raise TypeError("labels must be a pandas Series.")
    n_samples, n_features = expression.shape
    if n_samples == 0 or n_features == 0:
        raise ValueError("Expression matrix must have at least one sample and one feature.")
    if len(labels) != n_samples:
        raise ValueError("Expression rows and labels must have the same number of samples.")
    if variance_threshold < 0 or not np.isfinite(variance_threshold):
        raise ValueError("variance_threshold must be a finite, non-negative number.")
    if max_correlation_features < 2:
        raise ValueError("max_correlation_features must be at least 2.")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in expression.dtypes):
        raise TypeError("All expression columns must have numeric dtypes.")
    if not expression.columns.is_unique:
        raise ValueError("Expression feature names must be unique.")

    values = expression.to_numpy(dtype=float, copy=False, na_value=np.nan)
    missing_count = int(pd.isna(values).sum())
    nonfinite_count = int((~np.isfinite(values) & ~pd.isna(values)).sum())
    clean_expression = expression.replace([np.inf, -np.inf], np.nan)
    variance = clean_expression.var(axis=0, skipna=True).replace([np.inf, -np.inf], np.nan)
    near_zero_count = int((variance.dropna() <= variance_threshold).sum())
    zero_count = int(((values == 0) & np.isfinite(values)).sum())
    observed_count = int(np.isfinite(values).sum())

    counts = labels.value_counts(dropna=True, sort=False)
    class_counts: dict[str, int] = {}
    for label, count in counts.items():
        # JSON object keys must be strings. Preserve distinct label spellings.
        key = str(label)
        if key in class_counts:
            raise ValueError(f"Distinct target labels collide after string conversion: {key!r}.")
        class_counts[key] = int(count)

    smallest_class = int(counts.min()) if not counts.empty else 0
    largest_class = int(counts.max()) if not counts.empty else 0
    imbalance_ratio = (
        float(largest_class / smallest_class) if smallest_class else None
    )

    result: dict[str, Any] = {
        "dataset_id": dataset_id,
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "p_over_n": float(n_features / n_samples),
        "is_high_dimensional": bool(n_features > n_samples),
        "missing_value_count": missing_count,
        "nonfinite_value_count": nonfinite_count,
        "near_zero_variance_feature_count": near_zero_count,
        "zero_value_fraction": float(zero_count / observed_count) if observed_count else None,
        "target_missing_count": int(labels.isna().sum()),
        "class_counts": class_counts,
        "class_imbalance_ratio": imbalance_ratio,
        "correlation_summary": _correlation_summary(
            expression,
            max_features=max_correlation_features,
        ),
    }
    return result


def profile_to_json(profile: dict[str, Any], *, indent: int = 2) -> str:
    """Serialize a profile using strict JSON (no NaN or Infinity tokens)."""
    import json

    return json.dumps(profile, indent=indent, allow_nan=False, sort_keys=True)
