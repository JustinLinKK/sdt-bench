from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from clab.schemas.episode import EpisodeBudget, EpisodeSpec
from clab.schemas.metrics import EvaluationMetrics
from clab.schemas.patch import PatchProposal, PatchResult, ReviewResult
from clab.schemas.repo import RepoSpec
from clab.schemas.retrieval import RetrievalTrace, RetrievedChunk


class AgentContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode: EpisodeSpec
    repo_spec: RepoSpec
    workspace: str = Field(min_length=1)
    run_dir: str = Field(min_length=1)
    task_prompt: str = Field(min_length=1)
    visible_failure_signal: str = Field(min_length=1)
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    budget: EpisodeBudget
    backend_name: str = Field(min_length=1)


class AgentPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = ""
    steps: list[str] = Field(default_factory=list)


class IngestionDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ingest_visible_docs: bool = False
    reason: str = ""


class RetrievalDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = ""
    top_k: int = Field(default=0, ge=0)
    reason: str = ""


class EvaluationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: str = Field(min_length=1)
    repo_name: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    retrieval_trace: RetrievalTrace
    patch_proposal: PatchProposal
    patch_result: PatchResult
    review_result: ReviewResult
    metrics: EvaluationMetrics
    mutation_summary: dict[str, int | float]
