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
    SelectionCandidateResult,
)
from .tool_registry import FeatureSelectionToolRegistry

AgentRunner = Callable[[dict[str, Any]], Any]
SelectionExecutor = Callable[[list[FeatureSelectionConfig]], list[SelectionCandidateResult]]
ContextResolver = Callable[[list[str]], list[ProbeGeneContext]]
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
    candidates = []
    for candidate in state.get("selection_candidates", [])[:5]:
        serialized = (
            candidate.model_dump(mode="json")
            if isinstance(candidate, SelectionCandidateResult)
            else SelectionCandidateResult.model_validate(candidate).model_dump(mode="json")
        )
        serialized["selected_feature_ids"] = serialized["selected_feature_ids"][:50]
        candidates.append(serialized)
    return {
        "role": role,
        "dataset_id": state.get("dataset_id", ""),
        "metadata_profile": _safe_profile(state.get("metadata_profile", {})),
        "n_features": _feature_count(state),
        "probe_id_sample": state.get("probe_ids", [])[:20],
        "gene_contexts": contexts,
        "proposed_configs": proposals,
        "selection_candidates": candidates,
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
    contexts = [
        context if isinstance(context, ProbeGeneContext) else ProbeGeneContext.model_validate(context)
        for context in state.get("gene_contexts", [])
    ]
    resolved_contexts = [context for context in contexts if context.status == "resolved"]
    pathway_contexts = [context for context in resolved_contexts if context.pathways]
    profile = state.get("metadata_profile", {})
    correlation_summary = profile.get("correlation_summary", {}) if isinstance(profile, Mapping) else {}
    mean_correlation = correlation_summary.get("mean_absolute_pairwise_correlation")

    candidates = state.get("selection_candidates", [])
    candidate_by_config = {
        (candidate.config.method, candidate.config.k): candidate
        for candidate in candidates
    }

    def proposal_rank(proposal: AgentProposal) -> tuple[float, float, int, float]:
        candidate = candidate_by_config.get((proposal.config.method, proposal.config.k))
        if candidate is None:
            return (0.0, 0.0, 0, -float(mean_correlation or 0.0))
        has_pathway_support = int(
            candidate.pathway_supported_feature_count > 0
            or (pathway_contexts and proposal.config.method == "hypergraph_centrality")
        )
        return (
            candidate.cv_balanced_accuracy,
            candidate.selection_stability,
            has_pathway_support,
            -float(candidate.mean_pairwise_correlation or 0.0),
        )

    proposals.sort(
        key=proposal_rank,
        reverse=True,
    )

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
    candidate_feature_ids = {
        feature_id
        for candidate in candidates
        for feature_id in candidate.selected_feature_ids
    }
    pathway_supported_ids = [
        context.probe_id
        for context in resolved_contexts
        if context.pathways and context.probe_id in candidate_feature_ids
    ]
    if not contexts and not candidates:
        summary += " No selector output or Bio-RAG context was provided."
    elif candidates and not pathway_contexts:
        summary += " Candidate feature sets are available, but no pathway evidence resolved."
    return BioRefinerOutput(
        proposals=refined[:10],
        biological_summary=summary,
        pathway_supported_feature_ids=pathway_supported_ids,
    )


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
    candidates = state.get("selection_candidates", [])
    if not candidates:
        feedback.append("Selector tools did not return measured candidate feature sets.")
    else:
        best_balanced = max(item.cv_balanced_accuracy for item in candidates)
        best_stability = max(item.selection_stability for item in candidates)
        if best_stability < 0.1:
            feedback.append("Candidate feature sets show low split-to-split stability.")
        if best_balanced < 0.5:
            feedback.append("No candidate exceeded chance-level balanced accuracy in this CV run.")
        if not any(item.pathway_count for item in candidates):
            feedback.append("No pathway memberships were available for the candidate sets.")
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
    candidates = state.get("selection_candidates", [])
    if candidates:
        candidate = max(
            candidates,
            key=lambda item: (
                item.cv_balanced_accuracy,
                item.cv_macro_f1,
                item.selection_stability,
                item.pathway_supported_feature_count,
                -(item.mean_pairwise_correlation or 0.0),
            ),
        )
        selected = next(
            (
                proposal
                for proposal in valid
                if (proposal.config.method, proposal.config.k)
                == (candidate.config.method, candidate.config.k)
            ),
            AgentProposal(config=candidate.config, rationale="Selected from measured CV evidence."),
        )
        evidence_note = (
            f"Measured nested-CV balanced accuracy {candidate.cv_balanced_accuracy:.3f}, "
            f"macro-F1 {candidate.cv_macro_f1:.3f}, and feature stability "
            f"{candidate.selection_stability:.3f}."
        )
        if candidate.pathway_count:
            evidence_note += f" The selected set maps to {candidate.pathway_count} pathways."
        decision = AgentDebateDecision(
            config=candidate.config,
            biological_rationale=f"{selected.rationale} {evidence_note}",
            confidence=min(0.95, max(0.35, candidate.cv_balanced_accuracy)),
            selected_feature_ids=candidate.selected_feature_ids,
            cv_accuracy=candidate.cv_accuracy,
            cv_balanced_accuracy=candidate.cv_balanced_accuracy,
            cv_macro_f1=candidate.cv_macro_f1,
            selection_stability=candidate.selection_stability,
        )
        challenge = challenger_output or ChallengerOutput()
        return JudgeOutput(
            decision=decision,
            needs_revision=challenge.requires_revision,
            feedback=challenge.feedback + challenge.objections,
        )

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
    selection_executor: SelectionExecutor | None = None,
    context_resolver: ContextResolver | None = None,
):
    """Build the Planner → Bio-Refiner → Challenger → Judge LangGraph.

    The optional local selection executor receives validated proposals while
    retaining expression data in its closure. It returns compact CV evidence
    and selected identifiers for later review. An optional context resolver
    maps those identifiers to supplied biological records. Neither callback's
    matrix or labels are added to AgentState or sent to model runners.
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
            "selection_candidates": [],
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
        n_features = _feature_count(state)
        proposals: list[AgentProposal] = []
        for proposal in output.proposals[:5]:
            config = proposal.config.model_copy(
                update={
                    "k": min(proposal.config.k, n_features)
                    if n_features is not None
                    else proposal.config.k,
                    # The nested-CV evaluator currently scores each selector
                    # without a separate prefilter; keep final fit consistent.
                    "two_stage_cascade": False,
                }
            )
            proposals.append(proposal.model_copy(update={"config": config}))
        normalized = PlannerOutput.model_validate(
            {**output.model_dump(), "proposals": proposals}
        )
        return {
            **update,
            "planner_output": normalized,
            "proposed_configs": [proposal.config for proposal in normalized.proposals],
        }

    def selector_tools_node(state: AgentState) -> dict[str, Any]:
        errors = list(state.get("errors", []))
        candidates: list[SelectionCandidateResult] = []
        if selection_executor is None:
            errors.append(
                "No local selection executor was configured; this run can recommend "
                "settings but cannot return selected features."
            )
        else:
            try:
                candidates = [
                    item if isinstance(item, SelectionCandidateResult)
                    else SelectionCandidateResult.model_validate(item)
                    for item in selection_executor(state.get("proposed_configs", []))
                ]
            except Exception as exc:  # noqa: BLE001 - local selector errors are surfaced in state.
                errors.append(f"Local selector execution failed ({type(exc).__name__}).")

        updates: dict[str, Any] = {
            "selection_candidates": candidates,
            "errors": errors,
        }
        if context_resolver and candidates:
            selected_ids = list(
                dict.fromkeys(
                    feature_id
                    for candidate in candidates
                    for feature_id in candidate.selected_feature_ids
                )
            )
            try:
                resolved = context_resolver(selected_ids)
                existing_ids = {
                    context.probe_id
                    for context in state.get("gene_contexts", [])
                    if isinstance(context, ProbeGeneContext)
                }
                updates["gene_contexts"] = list(state.get("gene_contexts", [])) + [
                    context for context in resolved if context.probe_id not in existing_ids
                ]
            except Exception as exc:  # noqa: BLE001 - Bio-RAG errors do not discard selector results.
                updates["errors"] = errors + [
                    f"Bio-RAG context resolution failed ({type(exc).__name__})."
                ]
        return updates

    def bio_refiner_node(state: AgentState) -> dict[str, Any]:
        update = refiner_node(state)
        if update.get("retry_node") == "bio_refiner":
            return update
        output: BioRefinerOutput = update["bio_refiner_output"]
        candidates = {
            feature_id
            for candidate in state.get("selection_candidates", [])
            for feature_id in candidate.selected_feature_ids
        }
        pathway_backed = {
            context.probe_id
            for context in state.get("gene_contexts", [])
            if isinstance(context, ProbeGeneContext)
            and context.status == "resolved"
            and context.pathways
        }
        valid_ids = [
            feature_id
            for feature_id in output.pathway_supported_feature_ids
            if feature_id in candidates and feature_id in pathway_backed
        ]
        if len(valid_ids) != len(output.pathway_supported_feature_ids):
            errors = list(update.get("errors", state.get("errors", [])))
            errors.append("Bio-Refiner feature claims were restricted to resolved pathway records.")
            update["errors"] = errors
        update["bio_refiner_output"] = output.model_copy(
            update={"pathway_supported_feature_ids": valid_ids}
        )
        return update

    def judge_node(state: AgentState) -> dict[str, Any]:
        update = raw_judge_node(state)
        if update.get("retry_node") == "judge":
            return update
        output: JudgeOutput = update["judge_output"]
        candidates = state.get("selection_candidates", [])
        if selection_executor is not None and not candidates:
            # Do not present a selector recommendation as an executed feature
            # selection when every local candidate failed.
            decision = output.decision.model_copy(
                update={
                    "biological_rationale": (
                        "No selector candidate completed, so this run cannot provide "
                        "an evidence-backed feature set. Review the local selector "
                        "errors before interpreting the proposed configuration."
                    ),
                    "confidence": 0.0,
                    "selected_feature_ids": [],
                    "cv_accuracy": None,
                    "cv_balanced_accuracy": None,
                    "cv_macro_f1": None,
                    "selection_stability": None,
                }
            )
            output = output.model_copy(update={"decision": decision, "needs_revision": False})
            update["judge_output"] = output
            update["final_decision"] = decision
            update["fallback_used"] = True
            update["status"] = "fallback"
            update["errors"] = list(update.get("errors", state.get("errors", []))) + [
                "No local feature-selection candidate completed; no selected feature list is available."
            ]
            return update
        if candidates:
            chosen = next(
                (
                    item
                    for item in candidates
                    if (item.config.method, item.config.k)
                    == (output.decision.config.method, output.decision.config.k)
                ),
                None,
            )
            if chosen is None:
                output = _default_judge(state, tools)
                update["fallback_used"] = True
                update["errors"] = list(update.get("errors", state.get("errors", []))) + [
                    (
                        "Judge selected a configuration with no executed feature set; "
                        "the evidence-based selector choice was used instead."
                    )
                ]
                chosen = next(
                    item
                    for item in candidates
                    if (item.config.method, item.config.k)
                    == (output.decision.config.method, output.decision.config.k)
                )
            decision = output.decision.model_copy(
                update={
                    "selected_feature_ids": chosen.selected_feature_ids,
                    "cv_accuracy": chosen.cv_accuracy,
                    "cv_balanced_accuracy": chosen.cv_balanced_accuracy,
                    "cv_macro_f1": chosen.cv_macro_f1,
                    "selection_stability": chosen.selection_stability,
                }
            )
            output = output.model_copy(update={"decision": decision})
            update["judge_output"] = output
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
        candidates = state.get("selection_candidates", [])
        if candidates:
            best = max(
                candidates,
                key=lambda item: (
                    item.cv_balanced_accuracy,
                    item.cv_macro_f1,
                    item.selection_stability,
                    item.pathway_supported_feature_count,
                    -(item.mean_pairwise_correlation or 0.0),
                ),
            )
            decision = AgentDebateDecision(
                config=best.config,
                biological_rationale=(
                    "Review rounds were exhausted. Selected the best executed candidate "
                    "by nested-CV balanced accuracy, then macro-F1, stability, and pathway support."
                ),
                confidence=0.25,
                selected_feature_ids=best.selected_feature_ids,
                cv_accuracy=best.cv_accuracy,
                cv_balanced_accuracy=best.cv_balanced_accuracy,
                cv_macro_f1=best.cv_macro_f1,
                selection_stability=best.selection_stability,
            )
            return {
                "final_decision": decision,
                "fallback_used": True,
                "status": "fallback",
                "feedback": state.get("feedback", [])
                + ["Judge revision limit reached; best measured selector evidence was used."],
            }
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
    graph.add_node("selector_tools", selector_tools_node)
    graph.add_node("bio_refiner", bio_refiner_node)
    graph.add_node("challenger", challenger_node)
    graph.add_node("judge", judge_node)
    graph.add_node("fallback", fallback_node)
    graph.add_edge(START, "validate_input")
    graph.add_edge("validate_input", "planner")
    graph.add_conditional_edges(
        "planner", after_role("planner", "selector_tools"),
        {"planner": "planner", "selector_tools": "selector_tools"},
    )
    graph.add_edge("selector_tools", "bio_refiner")
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
