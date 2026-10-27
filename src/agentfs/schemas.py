"""Validated data contracts shared by the Track B components and teammates."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt

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
