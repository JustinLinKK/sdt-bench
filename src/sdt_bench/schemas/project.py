from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ProjectSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: str = Field(min_length=1)
    product_name: str = Field(min_length=1)
    framework_name: str = Field(min_length=1)
    framework_package: str = Field(min_length=1)
    framework_repo_url: str = Field(min_length=1)
    language: str = Field(min_length=1)
    package_manager: str = Field(min_length=1)
    notes: str | None = None
