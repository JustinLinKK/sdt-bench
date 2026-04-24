from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StateEnvironmentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    install_command: str = Field(min_length=1)
    offline: bool = True
    requirements_lock_path: str | None = None
    wheelhouse_path: str | None = None
    docker_image: str | None = None
    visible_test_command: str | None = None

    @field_validator("requirements_lock_path", "wheelhouse_path")
    @classmethod
    def normalize_optional_paths(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().replace("\\", "/")
        return normalized or None


class VisibleDocumentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    available_at: str = Field(min_length=1)
    source: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("path")
    @classmethod
    def normalize_path(cls, value: str) -> str:
        normalized = value.strip().replace("\\", "/")
        if not normalized:
            raise ValueError("document paths must not be empty")
        return normalized


class DocumentManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    documents: list[VisibleDocumentSpec] = Field(default_factory=list)


class TemporalStateSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state_id: str = Field(min_length=1)
    timeline_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    timestamp: str = Field(min_length=1)
    source_ref: str | None = None
    python_version: str | None = None
    dependency_snapshot: dict[str, str] = Field(default_factory=dict)
    environment: StateEnvironmentSpec
    snapshot_root: str = "product_snapshot"
    docs_manifest_path: str = "docs/manifest.yaml"
    docs_root: str = "docs"
    tests_root: str = "tests/visible"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("snapshot_root", "docs_manifest_path", "docs_root", "tests_root")
    @classmethod
    def normalize_relative_paths(cls, value: str) -> str:
        normalized = value.strip().replace("\\", "/")
        if not normalized:
            raise ValueError("relative paths must not be empty")
        return normalized
