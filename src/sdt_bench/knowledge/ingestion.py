from __future__ import annotations

from pathlib import Path

from sdt_bench.knowledge.chunking import build_visible_doc_chunks
from sdt_bench.knowledge.mutation_log import derive_mutation_records
from sdt_bench.schemas.episode import EpisodeSpec
from sdt_bench.schemas.result import IngestionDecision
from sdt_bench.schemas.retrieval import Chunk, MutationRecord, VectorDBSnapshot
from sdt_bench.utils.hashing import sha256_text
from sdt_bench.utils.time import utc_timestamp
from sdt_bench.vectordb.base import VectorDBBackend


def ingest_visible_docs(
    *,
    episode_dir: Path,
    episode: EpisodeSpec,
    chunk_size: int,
    overlap: int,
) -> tuple[list[Chunk], list[MutationRecord], VectorDBSnapshot]:
    chunks = build_visible_doc_chunks(episode_dir, episode, chunk_size=chunk_size, overlap=overlap)
    mutations, delete_chunk_ids, upsert_chunks = derive_mutation_records(
        episode_id=episode.episode_id,
        old_chunks=[],
        new_chunks=chunks,
        reason="visible_doc_staging",
    )
    del delete_chunk_ids, upsert_chunks
    snapshot = VectorDBSnapshot(
        backend_name="staging",
        collection_name=episode.episode_id,
        chunk_count=len(chunks),
        document_count=len({chunk.document_id for chunk in chunks}),
        chunk_ids=[chunk.chunk_id for chunk in chunks],
        created_at=utc_timestamp(),
        chunks=chunks,
    )
    return chunks, mutations, snapshot


def apply_ingestion_decision(
    *,
    episode: EpisodeSpec,
    decision: IngestionDecision,
    candidate_chunks: list[Chunk],
    backend: VectorDBBackend,
) -> tuple[list[Chunk], list[MutationRecord], VectorDBSnapshot]:
    existing = backend.dump_state()
    selected_chunks = _select_candidate_chunks(
        decision=decision,
        candidate_chunks=candidate_chunks,
        acquisition_budget=episode.budget.acquisition_budget,
    )
    if decision.strategy == "none":
        return [], [], existing

    mutations: list[MutationRecord] = []
    if decision.strategy == "full_reindex":
        if existing.chunk_ids:
            backend.delete_chunks(existing.chunk_ids)
            mutations.extend(
                _delete_mutation(
                    episode_id=episode.episode_id,
                    chunk=chunk,
                    reason="agent_full_reindex_delete",
                )
                for chunk in existing.chunks
            )
        if selected_chunks:
            backend.upsert_chunks(selected_chunks)
            mutations.extend(
                _insert_mutation(
                    episode_id=episode.episode_id,
                    chunk=chunk,
                    reason="agent_full_reindex_insert",
                )
                for chunk in selected_chunks
            )
        return selected_chunks, mutations, backend.dump_state()

    derived, delete_chunk_ids, upsert_chunks = derive_mutation_records(
        episode_id=episode.episode_id,
        old_chunks=existing.chunks,
        new_chunks=selected_chunks,
        reason=f"agent_ingestion:{decision.strategy}",
    )
    if delete_chunk_ids:
        backend.delete_chunks(delete_chunk_ids)
    if upsert_chunks:
        backend.upsert_chunks(upsert_chunks)
    return selected_chunks, derived, backend.dump_state()


def _select_candidate_chunks(
    *,
    decision: IngestionDecision,
    candidate_chunks: list[Chunk],
    acquisition_budget: int,
) -> list[Chunk]:
    if decision.strategy == "all_visible":
        selected_paths = sorted({chunk.source_path for chunk in candidate_chunks})
    else:
        selected_paths = [path.replace("\\", "/") for path in decision.selected_visible_doc_paths]
    if acquisition_budget > 0:
        selected_paths = selected_paths[:acquisition_budget]
    if not selected_paths:
        return []
    selected_path_set = set(selected_paths)
    return [chunk for chunk in candidate_chunks if chunk.source_path in selected_path_set]


def _delete_mutation(*, episode_id: str, chunk: Chunk, reason: str) -> MutationRecord:
    return MutationRecord(
        record_id=sha256_text(f"{episode_id}:delete:{chunk.chunk_id}"),
        episode_id=episode_id,
        operation="delete",
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        source_path=chunk.source_path,
        old_hash=chunk.content_hash,
        new_hash=None,
        timestamp=utc_timestamp(),
        reason=reason,
    )


def _insert_mutation(*, episode_id: str, chunk: Chunk, reason: str) -> MutationRecord:
    return MutationRecord(
        record_id=sha256_text(f"{episode_id}:insert:{chunk.chunk_id}:{reason}"),
        episode_id=episode_id,
        operation="insert",
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        source_path=chunk.source_path,
        old_hash=None,
        new_hash=chunk.content_hash,
        timestamp=utc_timestamp(),
        reason=reason,
    )
