from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


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
    max_visible_docs: int = Field(default=10, ge=0)
    acquisition_budget: int = Field(default=0, ge=0)
    max_patch_attempts: int = Field(default=1, ge=1)


class ProgrammingEpisodeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: str = Field(min_length=1)
    timeline_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    from_state_id: str = Field(min_length=1)
    to_state_id: str = Field(min_length=1)
    event_id: str = Field(min_length=1)
    task_family: str = Field(min_length=1)
    task_prompt: str = Field(min_length=1)
    visible_failure_path: str = Field(min_length=1)
    hidden_test_command: str = Field(min_length=1)
    hidden_test_manifest: str = Field(min_length=1)
    expected_files_of_interest: list[str] = Field(default_factory=list)
    success_criteria: SuccessCriteria = Field(default_factory=SuccessCriteria)
    budget: EpisodeBudget = Field(default_factory=EpisodeBudget)
    notes: str | None = None

    @field_validator("visible_failure_path", "hidden_test_manifest")
    @classmethod
    def normalize_relative_path(cls, value: str) -> str:
        normalized = value.strip().replace("\\", "/")
        if not normalized:
            raise ValueError("relative paths must not be empty")
        return normalized


EpisodeSpec = ProgrammingEpisodeSpec
