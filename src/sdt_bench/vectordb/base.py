from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Protocol

from sdt_bench.schemas.retrieval import Chunk, MutationRecord, RetrievedChunk, VectorDBSnapshot

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def vectorize_text(text: str, dimensions: int = 256) -> list[float]:
    counts = Counter(tokenize(text))
    vector = [0.0] * dimensions
    for token, count in counts.items():
        index = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16) % dimensions
        vector[index] += float(count)
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=False))


class VectorDBBackend(Protocol):
    backend_name: str
    collection_name: str

    def upsert_chunks(self, chunks: list[Chunk]) -> list[MutationRecord]: ...

    def delete_chunks(self, chunk_ids: list[str]) -> list[MutationRecord]: ...

    def query(self, query: str, top_k: int) -> list[RetrievedChunk]: ...

    def get_chunk(self, chunk_id: str) -> Chunk | None: ...

    def dump_state(self) -> VectorDBSnapshot: ...
