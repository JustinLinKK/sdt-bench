from pathlib import Path

from sdt_bench.vectordb.base import VectorDBBackend
from sdt_bench.vectordb.in_memory_backend import InMemoryBackend
from sdt_bench.vectordb.qdrant_backend import QdrantBackend


def build_backend(name: str, storage_path: Path, collection_name: str, dimensions: int) -> VectorDBBackend:
    if name == "in_memory":
        return InMemoryBackend(collection_name=collection_name, dimensions=dimensions)
    if name == "qdrant":
        return QdrantBackend(storage_path=storage_path, collection_name=collection_name, dimensions=dimensions)
    raise ValueError(f"Unsupported backend: {name}")


__all__ = ["VectorDBBackend", "build_backend", "InMemoryBackend", "QdrantBackend"]
