from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

EventType = Literal["patch", "minor", "major", "security", "synthetic"]


class DependencyEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(min_length=1)
    dependency_name: str = Field(min_length=1)
    ecosystem: str = Field(min_length=1)
    old_version: str = Field(min_length=1)
    new_version: str = Field(min_length=1)
    event_type: EventType
    summary: str = Field(min_length=1)
    breaking_change_expected: bool
    visible_doc_paths: list[str] = Field(default_factory=list)
    gold_mutation_paths: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
