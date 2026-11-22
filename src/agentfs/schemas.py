"""Validated data contracts shared by the Track B components and teammates."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

SelectionMethod = Literal["hypergraph_centrality", "mrmr", "lasso"]
MappingStatus = Literal[
    "resolved",
    "unmapped_probe",
    "ambiguous_probe_mapping",
    "gene_not_found",
    "ambiguous_gene_mapping",
]


class StrictSchema(BaseModel):
    """Base schema that rejects unknown fields and cannot be mutated."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class FeatureSelectionConfig(StrictSchema):
    """Approved settings for a feature-selection run; never executable code."""

    method: SelectionMethod
    k: StrictInt = Field(ge=5, le=500)
    two_stage_cascade: bool = True
    knockoff_fdr: float = Field(default=0.10, ge=0.0, le=0.10)


class AgentDebateDecision(StrictSchema):
    """Structured Judge result for downstream selection and explanation."""

    config: FeatureSelectionConfig
    biological_rationale: str = Field(min_length=1, max_length=4000)
    confidence: float = Field(ge=0.0, le=1.0)
    selected_feature_ids: list[str] = Field(default_factory=list, max_length=500)
    cv_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    cv_balanced_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    cv_macro_f1: float | None = Field(default=None, ge=0.0, le=1.0)
    selection_stability: float | None = Field(default=None, ge=0.0, le=1.0)


class SelectionCandidateResult(StrictSchema):
    """Measured result from running one selector configuration locally."""

    config: FeatureSelectionConfig
    selected_feature_ids: list[str] = Field(min_length=1, max_length=500)
    cv_accuracy: float = Field(ge=0.0, le=1.0)
    cv_balanced_accuracy: float = Field(ge=0.0, le=1.0)
    cv_macro_f1: float = Field(ge=0.0, le=1.0)
    selection_stability: float = Field(ge=0.0, le=1.0)
    pathway_count: StrictInt = Field(default=0, ge=0)
    pathway_supported_feature_count: StrictInt = Field(default=0, ge=0)
    mean_pairwise_correlation: float | None = Field(default=None, ge=0.0, le=1.0)


class AgentProposal(StrictSchema):
    """One validated feature-selection proposal with its rationale."""

    config: FeatureSelectionConfig
    rationale: str = Field(min_length=1, max_length=2000)


class PlannerOutput(StrictSchema):
    proposals: list[AgentProposal] = Field(min_length=1, max_length=20)
    summary: str = Field(default="", max_length=2000)


class BioRefinerOutput(StrictSchema):
    proposals: list[AgentProposal] = Field(min_length=1, max_length=20)
    biological_summary: str = Field(min_length=1, max_length=3000)
    pathway_supported_feature_ids: list[str] = Field(default_factory=list, max_length=500)


class ChallengerOutput(StrictSchema):
    objections: list[str] = Field(default_factory=list, max_length=20)
    feedback: list[str] = Field(default_factory=list, max_length=20)
    requires_revision: bool = False


class JudgeOutput(StrictSchema):
    decision: AgentDebateDecision
    needs_revision: bool = False
    feedback: list[str] = Field(default_factory=list, max_length=20)


class PathwayRecord(StrictSchema):
    """Pathway result with database identifiers and a readable description."""

    pathway_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    source: Literal["KEGG"] = "KEGG"


class ProbeGeneContext(StrictSchema):
    """Bio-RAG output for one platform-specific microarray probe."""

    probe_id: str = Field(min_length=1)
    platform: str | None = None
    organism: str = "Homo sapiens"
    status: MappingStatus
    gene_symbols: list[str] = Field(default_factory=list)
    candidate_gene_ids: list[str] = Field(default_factory=list)
    candidate_gene_symbols: list[str] = Field(default_factory=list)
    gene_symbol: str | None = None
    ncbi_gene_id: str | None = None
    gene_summary: str | None = None
    kegg_gene_id: str | None = None
    pathways: list[PathwayRecord] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    retrieved_at: datetime
    detail: str | None = None


class AgentGraphInput(StrictSchema):
    """Validated, sample-safe input accepted by the four-agent graph."""

    dataset_id: str = Field(min_length=1, max_length=200)
    metadata_profile: dict[str, Any] = Field(default_factory=dict)
    probe_ids: list[str] = Field(default_factory=list, max_length=20_000)
    gene_contexts: list[ProbeGeneContext] = Field(default_factory=list, max_length=20_000)
    proposed_configs: list[FeatureSelectionConfig] = Field(default_factory=list, max_length=20)
    max_agent_retries: StrictInt = Field(default=2, ge=0, le=5)
    max_revision_rounds: StrictInt = Field(default=2, ge=0, le=5)

    @model_validator(mode="after")
    def require_selector_sized_dataset(self) -> AgentGraphInput:
        n_features = self.metadata_profile.get("n_features")
        if not isinstance(n_features, int) or n_features <= 0:
            n_features = len(self.probe_ids) or None
        if n_features is not None and n_features < 5:
            raise ValueError("The current feature-selection contracts require at least 5 features.")
        return self
