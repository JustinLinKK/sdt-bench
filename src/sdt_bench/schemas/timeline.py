from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MemoryMode = Literal["persistent", "reset"]


class TimelineSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timeline_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    initial_state_id: str = Field(min_length=1)
    state_ids: list[str] = Field(default_factory=list)
    episode_ids: list[str] = Field(default_factory=list)
    default_memory_mode: MemoryMode = "persistent"
    description: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("state_ids", "episode_ids")
    @classmethod
    def validate_non_empty_lists(cls, value: list[str]) -> list[str]:
        normalized = [item.strip() for item in value if item.strip()]
        if not normalized:
            raise ValueError("timeline lists must not be empty")
        return normalized
