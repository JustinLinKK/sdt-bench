from __future__ import annotations

from collections import defaultdict

from sdt_bench.schemas.retrieval import Chunk, MutationRecord, RetrievedChunk, VectorDBSnapshot
from sdt_bench.utils.time import utc_timestamp
from sdt_bench.vectordb.base import VectorDBBackend, cosine_similarity, vectorize_text


class InMemoryBackend(VectorDBBackend):
    def __init__(self, collection_name: str, dimensions: int = 256) -> None:
        self.backend_name = "in_memory"
        self.collection_name = collection_name
        self.dimensions = dimensions
        self._chunks: dict[str, Chunk] = {}
        self._vectors: dict[str, list[float]] = {}

    def upsert_chunks(self, chunks: list[Chunk]) -> list[MutationRecord]:
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk
            self._vectors[chunk.chunk_id] = vectorize_text(chunk.content, self.dimensions)
        return []

    def delete_chunks(self, chunk_ids: list[str]) -> list[MutationRecord]:
        for chunk_id in chunk_ids:
            self._chunks.pop(chunk_id, None)
            self._vectors.pop(chunk_id, None)
        return []

    def query(self, query: str, top_k: int) -> list[RetrievedChunk]:
        query_vector = vectorize_text(query, self.dimensions)
        scored = []
        for chunk_id, chunk in self._chunks.items():
            score = cosine_similarity(query_vector, self._vectors.get(chunk_id, []))
            scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                source_path=chunk.source_path,
                content=chunk.content,
                score=score,
                metadata=chunk.metadata,
            )
            for score, chunk in scored[:top_k]
        ]

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)

    def dump_state(self) -> VectorDBSnapshot:
        chunks = sorted(
            self._chunks.values(), key=lambda item: (item.source_path, item.chunk_index)
        )
        documents = defaultdict(set)
        for chunk in chunks:
            documents[chunk.document_id].add(chunk.chunk_id)
        return VectorDBSnapshot(
            backend_name=self.backend_name,
            collection_name=self.collection_name,
            chunk_count=len(chunks),
            document_count=len(documents),
            chunk_ids=[chunk.chunk_id for chunk in chunks],
            created_at=utc_timestamp(),
            chunks=chunks,
        )
