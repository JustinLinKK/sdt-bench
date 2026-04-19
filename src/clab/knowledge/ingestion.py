from __future__ import annotations

from pathlib import Path

from clab.knowledge.chunking import build_visible_doc_chunks
from clab.knowledge.mutation_log import derive_mutation_records
from clab.schemas.episode import EpisodeSpec
from clab.schemas.retrieval import Chunk, MutationRecord, VectorDBSnapshot
from clab.vectordb.base import VectorDBBackend


def ingest_visible_docs(
    *,
    episode_dir: Path,
    episode: EpisodeSpec,
    backend: VectorDBBackend,
    chunk_size: int,
    overlap: int,
) -> tuple[list[Chunk], list[MutationRecord], VectorDBSnapshot]:
    old_snapshot = backend.dump_state()
    chunks = build_visible_doc_chunks(episode_dir, episode, chunk_size=chunk_size, overlap=overlap)
    mutations, delete_chunk_ids, upsert_chunks = derive_mutation_records(
        episode_id=episode.episode_id,
        old_chunks=old_snapshot.chunks,
        new_chunks=chunks,
        reason="visible_doc_ingestion",
    )
    if delete_chunk_ids:
        backend.delete_chunks(delete_chunk_ids)
    if upsert_chunks:
        backend.upsert_chunks(upsert_chunks)
    snapshot = backend.dump_state()
    return chunks, mutations, snapshot
