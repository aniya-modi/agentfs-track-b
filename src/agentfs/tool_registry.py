"""Unified Tool Registry exposing validated feature-selection tools to LLM agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline

from .baseline_selectors import LassoSelector, MRMRSelector
from .hypergraph_centrality import HypergraphCentralitySelector
from .schemas import FeatureSelectionConfig


@dataclass(frozen=True)
class ToolSpec:
    """Metadata describing a registered feature-selection tool for LLM agents."""

    name: str
    description: str
    requires_hypergraph: bool
    default_params: dict[str, Any]


@dataclass(frozen=True)
class ToolExecutionResult:
    """Structured output returned after running a validated selector on training data."""

    method: str
    k_requested: int
    k_selected: int
    two_stage_cascade_applied: bool
    knockoff_fdr: float
    selected_indices: list[int]
    selected_feature_ids: list[str]
    X_train_selected: np.ndarray


class FeatureSelectionToolRegistry:
    """Registry mapping Pydantic ``FeatureSelectionConfig`` objects to estimators.

    Supports pluggable integration with Bhumika's two-stage univariate cascade +
    Knockoff FDR filter without blocking parallel development.
    """

    def __init__(
        self,
        H: Any | None = None,
        cascade_filter_factory: Callable[[float], BaseEstimator] | None = None,
    ):
        self.H = H
        self.cascade_filter_factory = cascade_filter_factory
        self._specs: dict[str, ToolSpec] = {
            "hypergraph_centrality": ToolSpec(
                name="hypergraph_centrality",
                description=(
                    "Ranks genes by combining fold-local ANOVA F-scores with spectral "
                    "random-walk centrality over the KEGG pathway hypergraph."
                ),
                requires_hypergraph=True,
                default_params={"spectral_weight": 0.4, "damping": 0.85},
            ),
            "mrmr": ToolSpec(
                name="mrmr",
                description=(
                    "Minimum Redundancy Maximum Relevance (mRMR) greedy selector balancing "
                    "class separation against pairwise feature correlation."
                ),
                requires_hypergraph=False,
                default_params={"candidate_pool": 500},
            ),
            "lasso": ToolSpec(
                name="lasso",
                description=(
                    "L1-regularized logistic regression selector retaining features with "
                    "the largest sparse weight magnitudes."
                ),
                requires_hypergraph=False,
                default_params={"C": 1.0},
            ),
        }

    def list_tools(self) -> list[dict[str, Any]]:
        """Return JSON-serializable tool definitions for LangGraph Planner/Judge prompts."""
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "requires_hypergraph": spec.requires_hypergraph,
                "available": (self.H is not None) if spec.requires_hypergraph else True,
                "default_params": spec.default_params,
            }
            for spec in self._specs.values()
        ]

    def build_selector(self, config: FeatureSelectionConfig) -> BaseEstimator:
        """Instantiate the requested selector from a validated ``FeatureSelectionConfig``."""
        if config.method == "hypergraph_centrality":
            if self.H is None:
                raise ValueError(
                    "Method 'hypergraph_centrality' requires pathway incidence matrix H."
                )
            return HypergraphCentralitySelector(H=self.H, top_k=config.k)
        if config.method == "mrmr":
            return MRMRSelector(top_k=config.k)
        if config.method == "lasso":
            return LassoSelector(top_k=config.k)
        raise ValueError(f"Unsupported selection method: {config.method}")

    def execute(
        self,
        config: FeatureSelectionConfig,
        X_train: Any,
        y_train: Any,
        feature_ids: Sequence[str] | None = None,
    ) -> ToolExecutionResult:
        """Fit and transform training data using the validated agent configuration."""
        X_arr = np.asarray(X_train, dtype=float)
        y_arr = np.asarray(y_train)
        n_features = X_arr.shape[1]

        if feature_ids is not None and len(feature_ids) != n_features:
            raise ValueError(
                f"Expected {n_features} feature_ids, received {len(feature_ids)}."
            )

        steps: list[tuple[str, Any]] = []
        cascade_applied = False

        # Optional Stage-1 cascade pre-filter (uses Bhumika's Knockoff filter when plugged in,
        # or falls back to a safe ANOVA pre-screen if two_stage_cascade=True before merge).
        if config.two_stage_cascade and n_features > config.k:
            cascade_applied = True
            if self.cascade_filter_factory is not None:
                steps.append(("cascade", clone(self.cascade_filter_factory(config.knockoff_fdr))))
            elif config.method != "hypergraph_centrality":
                # Keep full dimension when H is indexed over all original features
                pre_k = min(n_features, max(config.k * 4, 250))
                steps.append(("cascade", SelectKBest(score_func=f_classif, k=pre_k)))
            else:
                cascade_applied = False

        steps.append(("selector", self.build_selector(config)))
        pipeline = Pipeline(steps)
        X_selected = pipeline.fit_transform(X_arr, y_arr)

        # Map final selected indices back to original feature space
        selector_support = pipeline.named_steps["selector"].get_support(indices=True)
        if "cascade" in pipeline.named_steps:
            cascade_support = pipeline.named_steps["cascade"].get_support(indices=True)
            final_indices = cascade_support[selector_support]
        else:
            final_indices = selector_support

        final_indices_list = [int(i) for i in final_indices]
        selected_ids = (
            [str(feature_ids[i]) for i in final_indices_list]
            if feature_ids is not None
            else [f"feature_{i}" for i in final_indices_list]
        )

        return ToolExecutionResult(
            method=config.method,
            k_requested=config.k,
            k_selected=len(final_indices_list),
            two_stage_cascade_applied=cascade_applied,
            knockoff_fdr=config.knockoff_fdr,
            selected_indices=final_indices_list,
            selected_feature_ids=selected_ids,
            X_train_selected=X_selected,
        )
