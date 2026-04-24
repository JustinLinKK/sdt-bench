from __future__ import annotations

from pathlib import Path

from sdt_bench.schemas.episode import ProgrammingEpisodeSpec
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


def build_doc_chunks_from_directory(
    docs_root: Path,
    visible_doc_paths: list[str],
    *,
    chunk_size: int,
    overlap: int,
    version_tag: str | None = None,
    metadata: dict | None = None,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for relative_path in sorted(visible_doc_paths):
        source_path = relative_path.replace("\\", "/")
        document_id = stable_document_id(source_path)
        text = (docs_root / relative_path).read_text(encoding="utf-8")
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
                    version_tag=version_tag,
                    metadata=metadata or {},
                )
            )
    return chunks


def build_visible_doc_chunks(
    episode_dir: Path,
    episode: ProgrammingEpisodeSpec,
    *,
    chunk_size: int,
    overlap: int,
    visible_doc_paths: list[str] | None = None,
) -> list[Chunk]:
    selected_paths = visible_doc_paths or []
    return build_doc_chunks_from_directory(
        episode_dir,
        selected_paths,
        chunk_size=chunk_size,
        overlap=overlap,
        version_tag=None,
        metadata={
            "episode_id": episode.episode_id,
            "project_id": episode.project_id,
            "is_visible_doc": True,
        },
    )
