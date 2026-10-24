"""Validated configuration types for feature selection and agent decisions."""

from typing import Literal

from pydantic import BaseModel, Field

SelectionMethod = Literal["hypergraph_centrality", "mrmr", "lasso"]


class FeatureSelectionConfig(BaseModel):
    """Approved settings for a feature-selection run."""

    method: SelectionMethod
    k: int = Field(ge=5, le=500)
    two_stage_cascade: bool = True
    knockoff_fdr: float = Field(default=0.10, gt=0, le=0.10)


class AgentDebateDecision(BaseModel):
    """Structured final decision from the orchestration layer."""

    config: FeatureSelectionConfig
    rationale: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
