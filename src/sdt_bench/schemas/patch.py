from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PatchProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: str = Field(min_length=1)
    patch_text: str = ""
    citations_used: list[str] = Field(default_factory=list)
    summary: str = ""


class ReviewResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = ""
    concerns: list[str] = Field(default_factory=list)


class PatchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: str = Field(min_length=1)
    patch_text: str = ""
    applied: bool
    apply_error: str | None = None
    files_changed: int = Field(default=0, ge=0)
    lines_added: int = Field(default=0, ge=0)
    lines_removed: int = Field(default=0, ge=0)
    visible_test_status: bool | None = None
    hidden_test_status: bool | None = None
    review_summary: str = ""
    citations_used: list[str] = Field(default_factory=list)
