from __future__ import annotations

from clab.schemas.retrieval import Chunk
from clab.vectordb.in_memory_backend import InMemoryBackend


def test_in_memory_vectordb_upsert_and_query() -> None:
    backend = InMemoryBackend(collection_name="test")
    backend.upsert_chunks(
        [
            Chunk(
                chunk_id="chunk-1",
                document_id="doc-1",
                source_path="visible_docs/a.md",
                content="requests urllib3 compatibility",
                content_hash="hash-1",
                chunk_index=0,
                created_at="2026-01-01T00:00:00+00:00",
                version_tag="2.0.0",
                metadata={},
            )
        ]
    )
    hits = backend.query("urllib3 compatibility", top_k=1)
    assert hits[0].chunk_id == "chunk-1"
    snapshot = backend.dump_state()
    assert snapshot.chunk_count == 1
