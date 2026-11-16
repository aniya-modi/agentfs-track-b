"""Repeated nested-CV evaluation and feature-selection stability metrics."""

from __future__ import annotations

from itertools import combinations
from typing import Any, Callable

import numpy as np
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def kuncheva_stability_index(
    selected_sets: list[np.ndarray], n_total_features: int
) -> float:
    """Mean pairwise Kuncheva index for equal-size selected feature sets."""
    if n_total_features < 1:
        raise ValueError("n_total_features must be positive.")
    scores: list[float] = []
    for first, second in combinations(selected_sets, 2):
        first_set = set(map(int, first))
        second_set = set(map(int, second))
        k = len(first_set)
        if k == 0 or k >= n_total_features or len(second_set) != k:
            continue
        expected_overlap = k * k / float(n_total_features)
        denominator = k - expected_overlap
        if denominator > 0:
            observed_overlap = len(first_set & second_set)
            scores.append((observed_overlap - expected_overlap) / denominator)
    return float(np.mean(scores)) if scores else 0.0


def jaccard_stability_index(selected_sets: list[np.ndarray]) -> float:
    """Mean pairwise Jaccard similarity of the selected feature sets."""
    scores: list[float] = []
    for first, second in combinations(selected_sets, 2):
        first_set = set(map(int, first))
        second_set = set(map(int, second))
        union = first_set | second_set
        scores.append(len(first_set & second_set) / len(union) if union else 0.0)
    return float(np.mean(scores)) if scores else 1.0


def run_repeated_nested_cv(
    X: np.ndarray,
    y: np.ndarray,
    selector_factory: Callable[[], Any],
    param_grid: dict[str, list[Any]],
    n_splits: int = 5,
    n_repeats: int = 3,
    inner_splits: int = 3,
    random_state: int = 42,
) -> dict[str, Any]:
    """Evaluate a selector and logistic model with all fitting inside CV folds."""
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    if X_arr.ndim != 2:
        raise ValueError("X must be a two-dimensional samples-by-features matrix.")
    if y_arr.ndim != 1 or len(y_arr) != X_arr.shape[0]:
        raise ValueError("y must be one-dimensional and match the rows of X.")
    if n_splits < 2 or inner_splits < 2 or n_repeats < 1:
        raise ValueError("Use at least two folds and one outer repeat.")

    outer_cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )
    accuracies: list[float] = []
    balanced_accuracies: list[float] = []
    macro_f1s: list[float] = []
    selected_indices: list[np.ndarray] = []
    best_params_by_fold: list[dict[str, Any]] = []

    for train_idx, test_idx in outer_cv.split(X_arr, y_arr):
        X_train, X_test = X_arr[train_idx], X_arr[test_idx]
        y_train, y_test = y_arr[train_idx], y_arr[test_idx]
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("selector", clone(selector_factory())),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced", max_iter=1000, random_state=random_state
                    ),
                ),
            ]
        )
        inner_cv = StratifiedKFold(
            n_splits=inner_splits, shuffle=True, random_state=random_state
        )
        search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=inner_cv,
            scoring="balanced_accuracy",
            n_jobs=1,
            refit=True,
        ).fit(X_train, y_train)

        best_pipeline = search.best_estimator_
        predictions = best_pipeline.predict(X_test)
        accuracies.append(accuracy_score(y_test, predictions))
        balanced_accuracies.append(balanced_accuracy_score(y_test, predictions))
        macro_f1s.append(f1_score(y_test, predictions, average="macro"))
        selected_indices.append(
            best_pipeline.named_steps["selector"].get_support(indices=True)
        )
        best_params_by_fold.append(dict(search.best_params_))

    return {
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
        "mean_balanced_accuracy": float(np.mean(balanced_accuracies)),
        "std_balanced_accuracy": float(np.std(balanced_accuracies)),
        "mean_macro_f1": float(np.mean(macro_f1s)),
        "std_macro_f1": float(np.std(macro_f1s)),
        "jaccard_stability": jaccard_stability_index(selected_indices),
        "selected_indices": selected_indices,
        "best_params_by_fold": best_params_by_fold,
    }
