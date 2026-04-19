from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from sdt_bench.schemas.event import EventType


class ReleaseRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    package_name: str = Field(min_length=1)
    ecosystem: str = Field(min_length=1)
    version: str = Field(min_length=1)
    published_at: str | None = None
    advisories: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EventStreamRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(min_length=1)
    repo_name: str = Field(min_length=1)
    dependency_name: str = Field(min_length=1)
    ecosystem: str = Field(min_length=1)
    old_version: str = Field(min_length=1)
    new_version: str = Field(min_length=1)
    event_type: EventType
    published_at: str | None = None
    new_advisories: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SnapshotManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repo_name: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    requested_ref: str = Field(min_length=1)
    resolved_commit: str = Field(min_length=1)
    workspace_path: str = Field(min_length=1)
    created_at: str = Field(min_length=1)


class AggregateSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_runs: int = Field(ge=0)
    mean_final_score: float = 0.0
    mean_mutation_f1: float = 0.0
    mean_freshness_score: float = 0.0
    hidden_pass_rate: float = 0.0
    auac_per_repo: dict[str, float] = Field(default_factory=dict)
    regression_rate: float = 0.0
