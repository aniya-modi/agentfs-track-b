"""Summarize dataset structure without returning expression values."""

from __future__ import annotations

from typing import Any

import pandas as pd


def profile_dataset(
    expression: pd.DataFrame,
    labels: pd.Series,
    *,
    variance_threshold: float = 1e-8,
) -> dict[str, Any]:
    """Return geometry, missingness, near-zero variance, and class balance metadata.

    Rows in ``expression`` are samples and columns are features. The returned
    profile contains aggregate counts only; it never includes sample values.
    """
    if expression.empty:
        raise ValueError("Expression matrix must not be empty.")
    if len(expression) != len(labels):
        raise ValueError("Expression rows and labels must have the same number of samples.")
    if variance_threshold < 0:
        raise ValueError("variance_threshold must be non-negative.")

    n_samples, n_features = expression.shape
    class_counts = labels.value_counts(dropna=False)
    smallest_class = int(class_counts.min())
    largest_class = int(class_counts.max())

    return {
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "p_over_n": float(n_features / n_samples),
        "is_high_dimensional": bool(n_features > n_samples),
        "missing_value_count": int(expression.isna().sum().sum()),
        "near_zero_variance_feature_count": int(
            (expression.var(axis=0, skipna=True) <= variance_threshold).sum()
        ),
        "class_counts": {str(label): int(count) for label, count in class_counts.items()},
        "class_imbalance_ratio": float(largest_class / smallest_class),
    }
