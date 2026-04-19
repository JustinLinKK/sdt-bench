from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from clab.schemas.event import DependencyEvent


class SuccessCriteria(BaseModel):
    model_config = ConfigDict(extra="forbid")

    require_hidden_tests_passed: bool = True
    require_patch_applied: bool = True
    max_files_changed: int = Field(default=5, ge=0)
    max_lines_changed: int = Field(default=200, ge=0)


class EpisodeBudget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_runtime_seconds: int = Field(default=900, ge=1)
    retrieval_top_k: int = Field(default=5, ge=1)
    max_visible_docs: int = Field(default=10, ge=1)


class EpisodeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: str = Field(min_length=1)
    repo_name: str = Field(min_length=1)
    base_commit: str = Field(min_length=7)
    base_ref: str = Field(min_length=1)
    dependency_event: DependencyEvent
    task_family: str = Field(min_length=1)
    task_prompt: str = Field(min_length=1)
    visible_failure_signal: str = Field(min_length=1)
    visible_doc_paths: list[str] = Field(default_factory=list)
    hidden_test_command: str = Field(min_length=1)
    hidden_test_manifest: str = Field(min_length=1)
    gold_patch_path: str | None = None
    expected_files_of_interest: list[str] = Field(default_factory=list)
    success_criteria: SuccessCriteria = Field(default_factory=SuccessCriteria)
    budget: EpisodeBudget = Field(default_factory=EpisodeBudget)
    notes: str | None = None

    @field_validator("visible_doc_paths")
    @classmethod
    def validate_visible_doc_paths(cls, value: list[str]) -> list[str]:
        normalized = [item.strip() for item in value if item.strip()]
        if not normalized:
            raise ValueError("visible_doc_paths must contain at least one document")
        return normalized
