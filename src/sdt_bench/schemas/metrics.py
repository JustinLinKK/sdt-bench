from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class EvaluationMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hidden_tests_passed: bool = False
    visible_tests_passed: bool | None = None
    patch_applied: bool = False
    files_changed: int = Field(default=0, ge=0)
    lines_added: int = Field(default=0, ge=0)
    lines_removed: int = Field(default=0, ge=0)
    fresh_chunk_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    stale_chunk_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    required_fresh_chunks_retrieved: bool = False
    mutation_precision: float = Field(default=0.0, ge=0.0, le=1.0)
    mutation_recall: float = Field(default=0.0, ge=0.0, le=1.0)
    mutation_f1: float = Field(default=0.0, ge=0.0, le=1.0)
    citation_overlap: float = Field(default=0.0, ge=0.0, le=1.0)
    correctness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    freshness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    churn_score: float = Field(default=0.0, ge=0.0, le=1.0)
    final_score: float = Field(default=0.0, ge=0.0, le=1.0)
