"""Four-role LangGraph debate with validated outputs and bounded fallbacks.

Agent runners are injected callables that receive a metadata-only context and
return a Pydantic model or mapping. The built-in defaults are deterministic so
the graph can run before an LLM provider is configured. Expression matrices and
sample labels are deliberately not part of the graph state or runner context.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, TypeVar

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from .agent_state import AgentState
from .schemas import (
    AgentDebateDecision,
    AgentGraphInput,
    AgentProposal,
    BioRefinerOutput,
    ChallengerOutput,
    FeatureSelectionConfig,
    JudgeOutput,
    PlannerOutput,
    ProbeGeneContext,
)
from .tool_registry import FeatureSelectionToolRegistry

AgentRunner = Callable[[dict[str, Any]], Any]
OutputModel = TypeVar("OutputModel", bound=BaseModel)


@dataclass(frozen=True)
class AgentRunners:
    """Optional structured-output callables for the four agent roles.

    Each callable receives JSON-friendly metadata and should return an instance
    or mapping accepted by its role's Pydantic V2 output model. A missing runner
    uses the deterministic built-in role implementation.
    """

    planner: AgentRunner | None = None
    bio_refiner: AgentRunner | None = None
    challenger: AgentRunner | None = None
    judge: AgentRunner | None = None


def _tool_catalog(registry: FeatureSelectionToolRegistry) -> list[dict[str, Any]]:
    return [tool for tool in registry.list_tools() if tool["available"]]


def _safe_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only aggregate profiler fields that are appropriate for agent context."""
    allowed = {
        "n_samples",
        "n_features",
        "p_over_n",
        "is_high_dimensional",
        "missing_value_count",
        "nonfinite_value_count",
        "near_zero_variance_feature_count",
        "zero_value_fraction",
        "target_missing_count",
        "class_counts",
        "class_imbalance_ratio",
    }
    safe = {key: value for key, value in profile.items() if key in allowed}
    if isinstance(profile.get("correlation_summary"), Mapping):
        correlation = profile["correlation_summary"]
        correlation_fields = {
            "features_used",
            "feature_limit",
            "mean_absolute_pairwise_correlation",
            "max_absolute_pairwise_correlation",
        }
        safe["correlation_summary"] = {
            key: value for key, value in correlation.items() if key in correlation_fields
        }
    return safe


def _state_context(
    state: AgentState, registry: FeatureSelectionToolRegistry, role: str
) -> dict[str, Any]:
    """Expose only bounded, non-expression context to an agent runner."""
    contexts = [
        context.model_dump(mode="json")
        if isinstance(context, ProbeGeneContext)
        else ProbeGeneContext.model_validate(context).model_dump(mode="json")
        for context in state.get("gene_contexts", [])[:50]
    ]
    for context in contexts:
        if context.get("gene_summary"):
            context["gene_summary"] = context["gene_summary"][:500]
        context["pathways"] = context.get("pathways", [])[:5]
        context["candidate_gene_ids"] = context.get("candidate_gene_ids", [])[:10]
        context["candidate_gene_symbols"] = context.get("candidate_gene_symbols", [])[:10]
    proposals = [
        proposal.model_dump(mode="json")
        for proposal in state.get("proposed_configs", [])
    ]
    return {
        "role": role,
        "dataset_id": state.get("dataset_id", ""),
        "metadata_profile": _safe_profile(state.get("metadata_profile", {})),
        "n_features": _feature_count(state),
        "probe_id_sample": state.get("probe_ids", [])[:20],
        "gene_contexts": contexts,
        "proposed_configs": proposals,
        "tool_catalog": _tool_catalog(registry),
        "feedback": state.get("feedback", []),
        "retry_count": state.get("revision_count", 0),
        "planner_output": (
            state["planner_output"].model_dump(mode="json")
            if state.get("planner_output") else None
        ),
        "bio_refiner_output": (
            state["bio_refiner_output"].model_dump(mode="json")
            if state.get("bio_refiner_output") else None
        ),
        "challenger_output": (
            state["challenger_output"].model_dump(mode="json")
            if state.get("challenger_output") else None
        ),
    }


def _feature_count(state: AgentState) -> int | None:
    profile = state.get("metadata_profile", {})
    value = profile.get("n_features") if isinstance(profile, Mapping) else None
    if isinstance(value, int) and value > 0:
        return value
    probe_ids = state.get("probe_ids", [])
    return len(probe_ids) if probe_ids else None


def _default_planner(
    state: AgentState, registry: FeatureSelectionToolRegistry
) -> PlannerOutput:
    available = {tool["name"] for tool in _tool_catalog(registry)}
    input_configs = state.get("proposed_configs", [])
    if input_configs:
        configs = [
            config if isinstance(config, FeatureSelectionConfig)
            else FeatureSelectionConfig.model_validate(config)
            for config in input_configs
        ]
        configs = [config for config in configs if config.method in available]
        n_features = _feature_count(state)
        if n_features is not None and n_features >= 5:
            configs = [
                config.model_copy(update={"k": n_features})
                if config.k > n_features >= 5
                else config
                for config in configs
            ]
    else:
        n_features = _feature_count(state)
        k = min(50, n_features) if n_features is not None else 50
        k = max(5, k)
        configs = [
            FeatureSelectionConfig(method=method, k=k, two_stage_cascade=False)
            for method in ("hypergraph_centrality", "mrmr", "lasso")
            if method in available
        ]

    if not configs:
        configs = [FeatureSelectionConfig(method="mrmr", k=5, two_stage_cascade=False)]
    proposals = [
        AgentProposal(
            config=config,
            rationale=(
                f"Propose {config.method} with k={config.k}; this is a candidate for "
                "review, not a diagnostic conclusion."
            ),
        )
        for config in configs[:10]
    ]
    return PlannerOutput(
        proposals=proposals,
        summary=f"Prepared {len(proposals)} proposals from the available selector tools.",
    )


def _default_bio_refiner(
    state: AgentState, registry: FeatureSelectionToolRegistry
) -> BioRefinerOutput:
    planner_output = state.get("planner_output")
    proposals = list(planner_output.proposals) if planner_output else []
    available = {tool["name"] for tool in _tool_catalog(registry)}
    contexts = [
        context if isinstance(context, ProbeGeneContext) else ProbeGeneContext.model_validate(context)
        for context in state.get("gene_contexts", [])
    ]
    resolved_contexts = [context for context in contexts if context.status == "resolved"]
    pathway_contexts = [context for context in resolved_contexts if context.pathways]
    profile = state.get("metadata_profile", {})
    correlation_summary = profile.get("correlation_summary", {}) if isinstance(profile, Mapping) else {}
    mean_correlation = correlation_summary.get("mean_absolute_pairwise_correlation")

    if pathway_contexts and "hypergraph_centrality" in available:
        proposals.sort(key=lambda proposal: proposal.config.method != "hypergraph_centrality")
    elif isinstance(mean_correlation, (int, float)) and mean_correlation >= 0.7:
        proposals.sort(key=lambda proposal: proposal.config.method != "mrmr")

    summary = (
        f"{len(resolved_contexts)} of {len(contexts)} supplied probe contexts resolved; "
        f"{len(pathway_contexts)} include pathway records."
    )
    refined = [
        AgentProposal(
            config=proposal.config,
            rationale=f"{proposal.rationale} Biological context: {summary}",
        )
        for proposal in proposals
    ]
    if not refined:
        return _default_bio_refiner(
            {**state, "planner_output": _default_planner(state, registry)}, registry
        )
    return BioRefinerOutput(proposals=refined[:10], biological_summary=summary)


def _default_challenger(
    state: AgentState, registry: FeatureSelectionToolRegistry
) -> ChallengerOutput:
    output = state.get("bio_refiner_output")
    proposals = list(output.proposals) if output else []
    available = {tool["name"] for tool in _tool_catalog(registry)}
    n_features = _feature_count(state)
    objections: list[str] = []
    feedback: list[str] = []

    if not proposals:
        objections.append("No validated selector proposals were supplied.")
    for proposal in proposals:
        config = proposal.config
        if config.method not in available:
            objections.append(f"Selector {config.method!r} is unavailable in this run.")
        if n_features is not None and config.k > n_features:
            objections.append(
                f"Requested k={config.k} exceeds the {n_features} available features."
            )
    if not state.get("gene_contexts"):
        feedback.append("No Bio-RAG contexts were supplied; biological support is limited.")
    elif not any(
        context.status == "resolved"
        for context in (
            item if isinstance(item, ProbeGeneContext) else ProbeGeneContext.model_validate(item)
            for item in state.get("gene_contexts", [])
        )
    ):
        feedback.append("No supplied probe context resolved; avoid gene-specific conclusions.")
    if not state.get("metadata_profile"):
        feedback.append("No dataset quality profile was supplied for review.")
    return ChallengerOutput(
        objections=objections,
        feedback=feedback,
        requires_revision=bool(objections),
    )


def _default_judge(
    state: AgentState, registry: FeatureSelectionToolRegistry
) -> JudgeOutput:
    refiner_output = state.get("bio_refiner_output")
    challenger_output = state.get("challenger_output")
    proposals = list(refiner_output.proposals) if refiner_output else []
    available = {tool["name"] for tool in _tool_catalog(registry)}
    valid = [proposal for proposal in proposals if proposal.config.method in available]
    if not valid:
        valid = list(_default_planner(state, registry).proposals)

    contexts = [
        context if isinstance(context, ProbeGeneContext) else ProbeGeneContext.model_validate(context)
        for context in state.get("gene_contexts", [])
    ]
    has_pathway_evidence = any(
        context.status == "resolved" and context.pathways for context in contexts
    )
    profile = state.get("metadata_profile", {})
    summary = profile.get("correlation_summary", {}) if isinstance(profile, Mapping) else {}
    correlation = summary.get("mean_absolute_pairwise_correlation")
    if has_pathway_evidence and "hypergraph_centrality" in available:
        preference = {"hypergraph_centrality": 0, "mrmr": 1, "lasso": 2}
    elif isinstance(correlation, (int, float)) and correlation >= 0.7:
        preference = {"mrmr": 0, "hypergraph_centrality": 1, "lasso": 2}
    else:
        preference = {"mrmr": 0, "lasso": 1, "hypergraph_centrality": 2}
    selected = min(valid, key=lambda proposal: preference.get(proposal.config.method, 99))

    challenge = challenger_output or ChallengerOutput()
    evidence_note = (
        "Resolved pathway context was available for review."
        if has_pathway_evidence
        else "No resolved pathway context was available; the decision relies on metadata and tool constraints."
    )
    rationale = f"{selected.rationale} Judge review: {evidence_note}"
    confidence = 0.72 if has_pathway_evidence and not challenge.objections else 0.55
    decision = AgentDebateDecision(
        config=selected.config,
        biological_rationale=rationale,
        confidence=confidence,
    )
    return JudgeOutput(
        decision=decision,
        needs_revision=challenge.requires_revision,
        feedback=challenge.feedback + challenge.objections,
    )


def _invoke_runner(runner: AgentRunner, context: dict[str, Any]) -> Any:
    return runner(context)


def _agent_node(
    role: str,
    output_key: str,
    output_model: type[OutputModel],
    runner: AgentRunner | None,
    fallback: Callable[[AgentState], OutputModel],
    registry: FeatureSelectionToolRegistry,
) -> Callable[[AgentState], dict[str, Any]]:
    def node(state: AgentState) -> dict[str, Any]:
        if runner is None:
            output = output_model.model_validate(fallback(state))
            return {output_key: output, "retry_node": None}

        attempts = dict(state.get("agent_attempts", {}))
        errors = list(state.get("errors", []))
        max_agent_retries = state.get("max_agent_retries", 2)
        try:
            raw_output = _invoke_runner(runner, _state_context(state, registry, role))
            output = output_model.model_validate(raw_output)
            return {output_key: output, "agent_attempts": attempts, "retry_node": None}
        except Exception as exc:  # noqa: BLE001 - agent/provider failures enter bounded retry handling.
            attempts[role] = attempts.get(role, 0) + 1
            errors.append(f"{role} attempt {attempts[role]} failed ({type(exc).__name__}).")
            if attempts[role] <= max_agent_retries:
                return {
                    "agent_attempts": attempts,
                    "errors": errors,
                    "retry_node": role,
                }
            output = output_model.model_validate(fallback(state))
            return {
                output_key: output,
                "agent_attempts": attempts,
                "errors": errors,
                "retry_node": None,
                "fallback_used": True,
            }

    return node


def build_agent_graph(
    *,
    runners: AgentRunners | None = None,
    registry: FeatureSelectionToolRegistry | None = None,
):
    """Build the Planner → Bio-Refiner → Challenger → Judge LangGraph.

    Custom runners are retried according to ``max_agent_retries`` in the input
    state. Their outputs are validated with Pydantic V2 before being accepted.
    Judge revision requests loop back to Planner up to
    ``max_revision_rounds``; exhaustion invokes a deterministic low-confidence
    fallback decision.
    """
    agents = runners or AgentRunners()
    tools = registry or FeatureSelectionToolRegistry()
    planner_fallback = lambda state: _default_planner(state, tools)
    refiner_fallback = lambda state: _default_bio_refiner(state, tools)
    challenger_fallback = lambda state: _default_challenger(state, tools)
    judge_fallback = lambda state: _default_judge(state, tools)

    def validate_input(state: AgentState) -> dict[str, Any]:
        validated = AgentGraphInput.model_validate(state)
        return {
            "dataset_id": validated.dataset_id,
            "metadata_profile": validated.metadata_profile,
            "probe_ids": validated.probe_ids,
            "gene_contexts": validated.gene_contexts,
            "proposed_configs": validated.proposed_configs,
            "max_agent_retries": validated.max_agent_retries,
            "max_revision_rounds": validated.max_revision_rounds,
            "agent_attempts": {},
            "revision_count": 0,
            "retry_node": None,
            "feedback": [],
            "fallback_used": False,
            "status": "running",
        }

    raw_planner_node = _agent_node(
        "planner", "planner_output", PlannerOutput, agents.planner,
        planner_fallback, tools,
    )
    refiner_node = _agent_node(
        "bio_refiner", "bio_refiner_output", BioRefinerOutput, agents.bio_refiner,
        refiner_fallback, tools,
    )
    challenger_node = _agent_node(
        "challenger", "challenger_output", ChallengerOutput, agents.challenger,
        challenger_fallback, tools,
    )
    raw_judge_node = _agent_node(
        "judge", "judge_output", JudgeOutput, agents.judge,
        judge_fallback, tools,
    )

    def planner_node(state: AgentState) -> dict[str, Any]:
        update = raw_planner_node(state)
        if update.get("retry_node") == "planner":
            return update
        output: PlannerOutput = update["planner_output"]
        return {
            **update,
            "proposed_configs": [proposal.config for proposal in output.proposals],
        }

    def judge_node(state: AgentState) -> dict[str, Any]:
        update = raw_judge_node(state)
        if update.get("retry_node") == "judge":
            return update
        output: JudgeOutput = update["judge_output"]
        if output.needs_revision:
            revision_count = state.get("revision_count", 0) + 1
            return {
                **update,
                "revision_count": revision_count,
                "feedback": output.feedback,
                "final_decision": None,
                "status": "running",
            }
        return {
            **update,
            "feedback": output.feedback,
            "final_decision": output.decision,
            "status": "accepted",
        }

    def after_role(role: str, next_role: str) -> Callable[[AgentState], str]:
        def route(state: AgentState) -> str:
            return role if state.get("retry_node") == role else next_role

        return route

    def after_judge(state: AgentState) -> str:
        if state.get("retry_node") == "judge":
            return "judge"
        output = state.get("judge_output")
        if output and output.needs_revision:
            if state.get("revision_count", 0) <= state.get("max_revision_rounds", 0):
                return "planner"
            return "fallback"
        return END

    def fallback_node(state: AgentState) -> dict[str, Any]:
        proposals: list[AgentProposal] = []
        refined = state.get("bio_refiner_output")
        planned = state.get("planner_output")
        if refined:
            proposals.extend(refined.proposals)
        if not proposals and planned:
            proposals.extend(planned.proposals)
        available = {tool["name"] for tool in _tool_catalog(tools)}
        selected = next(
            (proposal for proposal in proposals if proposal.config.method in available),
            AgentProposal(
                config=FeatureSelectionConfig(method="mrmr", k=5, two_stage_cascade=False),
                rationale="Use the deterministic mRMR fallback because no reviewed proposal remained.",
            ),
        )
        decision = AgentDebateDecision(
            config=selected.config,
            biological_rationale=(
                "Automated review reached its retry limit. "
                f"Fallback selected {selected.config.method} with k={selected.config.k}. "
                "This is a research configuration, not a diagnosis."
            ),
            confidence=0.25,
        )
        return {
            "final_decision": decision,
            "fallback_used": True,
            "status": "fallback",
            "feedback": state.get("feedback", [])
            + ["Judge revision limit reached; deterministic fallback was applied."],
        }

    graph = StateGraph(AgentState)
    graph.add_node("validate_input", validate_input)
    graph.add_node("planner", planner_node)
    graph.add_node("bio_refiner", refiner_node)
    graph.add_node("challenger", challenger_node)
    graph.add_node("judge", judge_node)
    graph.add_node("fallback", fallback_node)
    graph.add_edge(START, "validate_input")
    graph.add_edge("validate_input", "planner")
    graph.add_conditional_edges(
        "planner", after_role("planner", "bio_refiner"),
        {"planner": "planner", "bio_refiner": "bio_refiner"},
    )
    graph.add_conditional_edges(
        "bio_refiner", after_role("bio_refiner", "challenger"),
        {"bio_refiner": "bio_refiner", "challenger": "challenger"},
    )
    graph.add_conditional_edges(
        "challenger", after_role("challenger", "judge"),
        {"challenger": "challenger", "judge": "judge"},
    )
    graph.add_conditional_edges(
        "judge", after_judge,
        {"judge": "judge", "planner": "planner", "fallback": "fallback", END: END},
    )
    graph.add_edge("fallback", END)
    return graph.compile()
