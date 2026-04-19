from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class RepoSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    github_url: str = Field(min_length=1)
    package_name: str | None = None
    ecosystem: str | None = None
    default_branch: str = Field(min_length=1)
    language: str = Field(min_length=1)
    package_manager: str = Field(min_length=1)
    install_command: str = Field(min_length=1)
    test_command: str = Field(min_length=1)
    dockerfile_path: str | None = None
    image_reference: str | None = None
    supported_dependency_files: list[str] = Field(default_factory=list)
    notes: str | None = None
