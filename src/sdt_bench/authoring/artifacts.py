from __future__ import annotations

from pathlib import Path

from sdt_bench.knowledge.chunking import build_doc_chunks_from_directory
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
    event_dir = _resolve_event_dir(episode_dir, episode)
    visible_doc_paths = _resolve_visible_doc_paths(event_dir)
    new_chunks = build_doc_chunks_from_directory(
        event_dir,
        visible_doc_paths,
        chunk_size=chunk_size,
        overlap=overlap,
        version_tag=episode.to_state_id,
        metadata={"episode_id": episode.episode_id, "project_id": episode.project_id},
    )
    old_chunks = _build_old_chunks(
        visible_doc_paths=visible_doc_paths,
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
    required_chunk_ids = [chunk.chunk_id for chunk in sorted(new_chunks, key=lambda item: (item.source_path, item.chunk_index))][
        :required_chunk_count
    ]
    write_yaml(
        event_dir / "artifacts" / "gold_mutations.yaml",
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
        event_dir / "artifacts" / "expected_retrieval_chunks.yaml",
        {"required_chunk_ids": required_chunk_ids},
    )
    return {
        "mutations": len(mutations),
        "required_chunk_ids": len(required_chunk_ids),
    }


def _build_old_chunks(
    *,
    visible_doc_paths: list[str],
    episode: EpisodeSpec,
    chunk_size: int,
    overlap: int,
    old_visible_doc_root: Path | None,
) -> list[Chunk]:
    if old_visible_doc_root is None:
        return []
    translated_paths = []
    for relative_path in visible_doc_paths:
        candidate = old_visible_doc_root / Path(relative_path).name
        if candidate.exists():
            translated_paths.append(str(candidate.relative_to(old_visible_doc_root)).replace("\\", "/"))
    if not translated_paths:
        return []
    return build_doc_chunks_from_directory(
        old_visible_doc_root,
        translated_paths,
        chunk_size=chunk_size,
        overlap=overlap,
        version_tag=episode.from_state_id,
        metadata={"episode_id": episode.episode_id, "project_id": episode.project_id},
    )


def _resolve_event_dir(episode_dir: Path, episode: EpisodeSpec) -> Path:
    project_root = episode_dir.parent.parent
    return project_root / "events" / episode.event_id


def _resolve_visible_doc_paths(event_dir: Path) -> list[str]:
    event_yaml = event_dir / "event.yaml"
    from sdt_bench.utils.fs import read_yaml

    payload = read_yaml(event_yaml)
    return [item.replace("\\", "/") for item in payload.get("visible_doc_paths", [])]
