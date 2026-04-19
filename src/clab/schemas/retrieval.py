from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

MutationOperation = Literal["insert", "update", "delete", "tombstone"]


class Chunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    content: str
    content_hash: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)
    created_at: str = Field(min_length=1)
    version_tag: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    content: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class MutationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    operation: MutationOperation
    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    old_hash: str | None = None
    new_hash: str | None = None
    timestamp: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class RetrievalTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: str = Field(min_length=1)
    query: str
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    retrieved_document_ids: list[str] = Field(default_factory=list)
    scores: list[float] = Field(default_factory=list)
    freshness_labels: list[str] = Field(default_factory=list)
    timestamp: str = Field(min_length=1)


class VectorDBSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend_name: str = Field(min_length=1)
    collection_name: str = Field(min_length=1)
    chunk_count: int = Field(ge=0)
    document_count: int = Field(ge=0)
    chunk_ids: list[str] = Field(default_factory=list)
    created_at: str = Field(min_length=1)
    chunks: list[Chunk] = Field(default_factory=list)
