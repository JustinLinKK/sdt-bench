from __future__ import annotations

from sdt_bench.knowledge.mutation_log import derive_mutation_records
from sdt_bench.schemas.retrieval import Chunk


def make_chunk(chunk_id: str, content_hash: str, index: int) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc",
        source_path="visible_docs/a.md",
        content=f"chunk-{index}",
        content_hash=content_hash,
        chunk_index=index,
        created_at="2026-01-01T00:00:00+00:00",
        version_tag="1.0.0",
        metadata={},
    )


def test_mutation_log_detects_insert_update_and_tombstone() -> None:
    old = [make_chunk("old0", "hash0", 0), make_chunk("old1", "hash1", 1)]
    new = [make_chunk("new0", "hash-new0", 0), make_chunk("new2", "hash2", 2)]
    mutations, deletes, upserts = derive_mutation_records(
        episode_id="episode",
        old_chunks=old,
        new_chunks=new,
        reason="test",
    )
    operations = {mutation.operation for mutation in mutations}
    assert "update" in operations
    assert "insert" in operations
    assert "tombstone" in operations
    assert deletes == ["old0", "old1"]
    assert [chunk.chunk_id for chunk in upserts] == ["new0", "new2"]
