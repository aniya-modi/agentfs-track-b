"""Shared state shape for future LangGraph orchestration."""

from typing import Any, TypedDict

from .schemas import AgentDebateDecision


class AgentState(TypedDict, total=False):
    metadata_profile: dict[str, Any]
    gene_mappings: list[dict[str, Any]]
    proposed_configs: list[dict[str, Any]]
    final_decision: AgentDebateDecision
    errors: list[str]
