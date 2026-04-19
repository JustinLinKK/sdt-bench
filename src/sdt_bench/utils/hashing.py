from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_document_id(source_path: str) -> str:
    return sha256_text(source_path)


def stable_chunk_id(document_id: str, chunk_index: int, content_hash: str) -> str:
    return sha256_text(f"{document_id}:{chunk_index}:{content_hash}")
