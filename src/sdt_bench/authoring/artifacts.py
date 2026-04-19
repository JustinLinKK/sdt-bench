from __future__ import annotations

from pathlib import Path

from sdt_bench.knowledge.chunking import build_visible_doc_chunks
from sdt_bench.knowledge.mutation_log import derive_mutation_records
from sdt_bench.schemas import Chunk, EpisodeSpec
from sdt_bench.utils.fs import write_yaml


def synthesize_episode_artifacts(
    *,
    episode_dir: Path,
    episode: EpisodeSpec,
    chunk_size: int,
    overlap: int,
    old_visible_doc_root: Path | None = None,
    required_chunk_count: int = 2,
) -> dict[str, int]:
    new_chunks = build_visible_doc_chunks(
        episode_dir,
        episode,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    old_chunks = _build_old_chunks(
        episode_dir=episode_dir,
        episode=episode,
        chunk_size=chunk_size,
        overlap=overlap,
        old_visible_doc_root=old_visible_doc_root,
    )
    mutations, _, _ = derive_mutation_records(
        episode_id=episode.episode_id,
        old_chunks=old_chunks,
        new_chunks=new_chunks,
        reason="artifact_synthesis",
    )
    required_chunk_ids = [
        chunk.chunk_id
        for chunk in sorted(new_chunks, key=lambda item: (item.source_path, item.chunk_index))
    ][:required_chunk_count]
    write_yaml(
        episode_dir / "artifacts" / "gold_mutations.yaml",
        {
            "mutations": [
                {
                    "operation": mutation.operation,
                    "chunk_id": mutation.chunk_id,
                    "source_path": mutation.source_path,
                }
                for mutation in mutations
            ]
        },
    )
    write_yaml(
        episode_dir / "artifacts" / "expected_retrieval_chunks.yaml",
        {"required_chunk_ids": required_chunk_ids},
    )
    return {
        "mutations": len(mutations),
        "required_chunk_ids": len(required_chunk_ids),
    }


def _build_old_chunks(
    *,
    episode_dir: Path,
    episode: EpisodeSpec,
    chunk_size: int,
    overlap: int,
    old_visible_doc_root: Path | None,
) -> list[Chunk]:
    if old_visible_doc_root is None:
        return []
    translated_paths = []
    for relative_path in episode.visible_doc_paths:
        candidate = old_visible_doc_root / Path(relative_path).name
        if candidate.exists():
            translated_paths.append(str(candidate.relative_to(old_visible_doc_root)).replace("\\", "/"))
    if not translated_paths:
        return []

    temp_episode = episode.model_copy(deep=True)
    temp_episode.visible_doc_paths = translated_paths
    return build_visible_doc_chunks(
        old_visible_doc_root,
        temp_episode,
        chunk_size=chunk_size,
        overlap=overlap,
        visible_doc_paths=translated_paths,
    )
