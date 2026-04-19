from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from sdt_bench.schemas.episode import EpisodeBudget, ProgrammingEpisodeSpec
from sdt_bench.schemas.event import DependencyEventSpec
from sdt_bench.schemas.io import MemoryManifest, StepInputManifest
from sdt_bench.schemas.metrics import EvaluationMetrics
from sdt_bench.schemas.patch import PatchProposal, PatchResult, ReviewResult
from sdt_bench.schemas.repo import RepoSpec
from sdt_bench.schemas.retrieval import RetrievalTrace, RetrievedChunk
from sdt_bench.schemas.state import TemporalStateSpec
from sdt_bench.schemas.timeline import MemoryMode, TimelineSpec


class AgentContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timeline: TimelineSpec
    episode: ProgrammingEpisodeSpec
    event: DependencyEventSpec
    from_state: TemporalStateSpec
    to_state: TemporalStateSpec
    repo_spec: RepoSpec
    step_manifest: StepInputManifest
    step_index: int = Field(ge=0)
    workspace: str = Field(min_length=1)
    input_dir: str = Field(min_length=1)
    output_dir: str = Field(min_length=1)
    run_dir: str = Field(min_length=1)
    task_prompt: str = Field(min_length=1)
    visible_failure_signal: str = Field(min_length=1)
    available_visible_doc_paths: list[str] = Field(default_factory=list)
    docs_manifest_path: str = Field(min_length=1)
    memory_manifest: MemoryManifest
    memory_chunks_path: str = Field(min_length=1)
    visible_chunks_path: str | None = None
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    budget: EpisodeBudget
    backend_name: str = Field(min_length=1, default="in_memory")
    memory_mode: MemoryMode = "persistent"


class AgentPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = ""
    steps: list[str] = Field(default_factory=list)


class IngestionDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: str = "none"
    ingest_visible_docs: bool = False
    selected_visible_doc_paths: list[str] = Field(default_factory=list)
    acquisitions: list[str] = Field(default_factory=list)
    reason: str = ""


class RetrievalDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = ""
    top_k: int = Field(default=0, ge=0)
    reason: str = ""


class CommandResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: str = ""
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""
    passed: bool = False


class StepEvaluationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timeline_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    step_index: int = Field(ge=0)
    repo_name: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    agent_name: str = Field(min_length=1)
    memory_mode: MemoryMode = "persistent"
    retrieval_trace: RetrievalTrace
    patch_proposal: PatchProposal
    patch_result: PatchResult
    review_result: ReviewResult
    metrics: EvaluationMetrics
    mutation_summary: dict[str, int | float]
    visible_tests: CommandResult
    hidden_tests: CommandResult
    memory_manifest: MemoryManifest


class TimelineAggregateMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_count: int = Field(default=0, ge=0)
    hidden_pass_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    cumulative_success: float = Field(default=0.0, ge=0.0, le=1.0)
    adaptation_area: float = Field(default=0.0, ge=0.0, le=1.0)
    average_stale_retrieval_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    mean_time_to_recover: float = Field(default=0.0, ge=0.0)
    max_drawdown: float = Field(default=0.0, ge=0.0, le=1.0)


class TimelineEvaluationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timeline_id: str = Field(min_length=1)
    repo_name: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    agent_name: str = Field(min_length=1)
    memory_mode: MemoryMode = "persistent"
    steps: list[StepEvaluationResult] = Field(default_factory=list)
    aggregate: TimelineAggregateMetrics


EvaluationResult = StepEvaluationResult
