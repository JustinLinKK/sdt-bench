from __future__ import annotations

from pathlib import Path

from sdt_bench.schemas.episode import EpisodeSpec
from sdt_bench.schemas.retrieval import Chunk
from sdt_bench.utils.hashing import sha256_text, stable_chunk_id, stable_document_id
from sdt_bench.utils.time import utc_timestamp


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if not text:
        return []
    windows: list[str] = []
    start = 0
    stride = max(1, chunk_size - overlap)
    while start < len(text):
        windows.append(text[start : start + chunk_size])
        start += stride
    return windows


def build_visible_doc_chunks(
    episode_dir: Path,
    episode: EpisodeSpec,
    *,
    chunk_size: int,
    overlap: int,
    visible_doc_paths: list[str] | None = None,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for relative_path in sorted(visible_doc_paths or episode.visible_doc_paths):
        source_path = relative_path.replace("\\", "/")
        document_id = stable_document_id(source_path)
        text = (episode_dir / relative_path).read_text(encoding="utf-8")
        for chunk_index, content in enumerate(chunk_text(text, chunk_size, overlap)):
            content_hash = sha256_text(content)
            chunks.append(
                Chunk(
                    chunk_id=stable_chunk_id(document_id, chunk_index, content_hash),
                    document_id=document_id,
                    source_path=source_path,
                    content=content,
                    content_hash=content_hash,
                    chunk_index=chunk_index,
                    created_at=utc_timestamp(),
                    version_tag=episode.dependency_event.new_version,
                    metadata={
                        "episode_id": episode.episode_id,
                        "repo_name": episode.repo_name,
                        "is_visible_doc": True,
                    },
                )
            )
    return chunks
