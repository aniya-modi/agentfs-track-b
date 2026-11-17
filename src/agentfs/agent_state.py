"""Typed shared state contract for the LangGraph agent workflow."""

from __future__ import annotations

from typing import Any, Literal, NotRequired, TypedDict

from .schemas import (
    AgentDebateDecision,
    BioRefinerOutput,
    ChallengerOutput,
    FeatureSelectionConfig,
    JudgeOutput,
    PlannerOutput,
    ProbeGeneContext,
)


class AgentState(TypedDict):
    """State passed between planner, refiner, challenger, and judge nodes.

    Values are metadata, biological identifiers/context, validated proposals,
    and review outputs only. Expression matrices and sample-level labels stay
    in the local analysis pipeline and are not part of agent state.
    """

    dataset_id: str
    metadata_profile: dict[str, Any]
    probe_ids: list[str]
    gene_contexts: list[ProbeGeneContext]
    proposed_configs: NotRequired[list[FeatureSelectionConfig]]
    planner_output: NotRequired[PlannerOutput]
    bio_refiner_output: NotRequired[BioRefinerOutput]
    challenger_output: NotRequired[ChallengerOutput]
    judge_output: NotRequired[JudgeOutput]
    final_decision: NotRequired[AgentDebateDecision | None]
    agent_attempts: NotRequired[dict[str, int]]
    revision_count: NotRequired[int]
    max_agent_retries: NotRequired[int]
    max_revision_rounds: NotRequired[int]
    retry_node: NotRequired[str | None]
    feedback: NotRequired[list[str]]
    fallback_used: NotRequired[bool]
    status: NotRequired[Literal["running", "accepted", "fallback"]]
    errors: NotRequired[list[str]]
