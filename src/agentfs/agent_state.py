"""Typed shared state contract for the future LangGraph workflow."""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict

from .schemas import AgentDebateDecision, FeatureSelectionConfig, ProbeGeneContext


class AgentState(TypedDict):
    """State passed between planner, refiner, challenger, and judge nodes.

    Values are metadata and biological identifiers/context only. The expression
    matrix remains in the local analysis pipeline and is not part of agent state.
    """

    dataset_id: str
    metadata_profile: dict[str, Any]
    probe_ids: list[str]
    gene_contexts: list[ProbeGeneContext]
    proposed_configs: list[FeatureSelectionConfig]
    final_decision: NotRequired[AgentDebateDecision]
    errors: NotRequired[list[str]]
