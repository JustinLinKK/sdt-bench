from __future__ import annotations

from collections import Counter
from pathlib import Path

from clab.schemas.retrieval import Chunk, MutationRecord
from clab.utils.fs import write_jsonl
from clab.utils.hashing import sha256_text
from clab.utils.time import utc_timestamp


def derive_mutation_records(
    *,
    episode_id: str,
    old_chunks: list[Chunk],
    new_chunks: list[Chunk],
    reason: str,
) -> tuple[list[MutationRecord], list[str], list[Chunk]]:
    old_map = {(chunk.document_id, chunk.chunk_index): chunk for chunk in old_chunks}
    new_map = {(chunk.document_id, chunk.chunk_index): chunk for chunk in new_chunks}
    mutations: list[MutationRecord] = []
    delete_chunk_ids: list[str] = []
    upsert_chunks: list[Chunk] = []
    timestamp = utc_timestamp()

    for key, new_chunk in new_map.items():
        old_chunk = old_map.get(key)
        if old_chunk is None:
            mutations.append(
                MutationRecord(
                    record_id=sha256_text(f"{episode_id}:insert:{new_chunk.chunk_id}"),
                    episode_id=episode_id,
                    operation="insert",
                    chunk_id=new_chunk.chunk_id,
                    document_id=new_chunk.document_id,
                    source_path=new_chunk.source_path,
                    old_hash=None,
                    new_hash=new_chunk.content_hash,
                    timestamp=timestamp,
                    reason=reason,
                )
            )
            upsert_chunks.append(new_chunk)
        elif old_chunk.content_hash != new_chunk.content_hash:
            delete_chunk_ids.append(old_chunk.chunk_id)
            mutations.append(
                MutationRecord(
                    record_id=sha256_text(f"{episode_id}:update:{new_chunk.chunk_id}"),
                    episode_id=episode_id,
                    operation="update",
                    chunk_id=new_chunk.chunk_id,
                    document_id=new_chunk.document_id,
                    source_path=new_chunk.source_path,
                    old_hash=old_chunk.content_hash,
                    new_hash=new_chunk.content_hash,
                    timestamp=timestamp,
                    reason=reason,
                )
            )
            upsert_chunks.append(new_chunk)

    for key, old_chunk in old_map.items():
        if key not in new_map:
            delete_chunk_ids.append(old_chunk.chunk_id)
            mutations.append(
                MutationRecord(
                    record_id=sha256_text(f"{episode_id}:tombstone:{old_chunk.chunk_id}"),
                    episode_id=episode_id,
                    operation="tombstone",
                    chunk_id=old_chunk.chunk_id,
                    document_id=old_chunk.document_id,
                    source_path=old_chunk.source_path,
                    old_hash=old_chunk.content_hash,
                    new_hash=None,
                    timestamp=timestamp,
                    reason=reason,
                )
            )

    return mutations, delete_chunk_ids, upsert_chunks


def write_mutation_log(path: Path, mutations: list[MutationRecord]) -> None:
    write_jsonl(path, [mutation.model_dump(mode="json") for mutation in mutations])


def summarize_mutations(mutations: list[MutationRecord]) -> dict[str, int]:
    counts = Counter(mutation.operation for mutation in mutations)
    return {
        "insert": counts.get("insert", 0),
        "update": counts.get("update", 0),
        "delete": counts.get("delete", 0),
        "tombstone": counts.get("tombstone", 0),
        "total": len(mutations),
    }
