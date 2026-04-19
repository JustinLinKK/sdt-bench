from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from sdt_bench.schemas.timeline import MemoryMode


class MemoryManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    snapshot_id: str = Field(min_length=1)
    source_episode_id: str | None = None
    chunk_count: int = Field(default=0, ge=0)
    document_count: int = Field(default=0, ge=0)
    persisted: bool = False


class StepInputManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timeline_id: str = Field(min_length=1)
    repo_name: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    step_index: int = Field(ge=0)
    agent_name: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    memory_mode: MemoryMode
    from_state_id: str = Field(min_length=1)
    to_state_id: str = Field(min_length=1)
    event_id: str = Field(min_length=1)
    available_at: str = Field(min_length=1)
    workspace: str = Field(min_length=1)
    input_dir: str = Field(min_length=1)
    output_dir: str = Field(min_length=1)
    available_visible_doc_paths: list[str] = Field(default_factory=list)
    visible_failure_path: str = Field(min_length=1)
    docs_manifest_path: str = Field(min_length=1)
    memory_manifest: MemoryManifest


class StepOutputManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: str = Field(min_length=1)
    plan_path: str = Field(min_length=1)
    ingestion_decision_path: str = Field(min_length=1)
    retrieval_decision_path: str = Field(min_length=1)
    retrieval_trace_path: str = Field(min_length=1)
    patch_path: str = Field(min_length=1)
    citations_path: str = Field(min_length=1)
    memory_mutations_path: str = Field(min_length=1)
    review_path: str = Field(min_length=1)
