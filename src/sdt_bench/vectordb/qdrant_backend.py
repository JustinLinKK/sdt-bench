from __future__ import annotations

from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from sdt_bench.schemas.retrieval import Chunk, MutationRecord, RetrievedChunk, VectorDBSnapshot
from sdt_bench.utils.fs import ensure_dir
from sdt_bench.utils.time import utc_timestamp
from sdt_bench.vectordb.base import VectorDBBackend, vectorize_text


def _point_id(chunk_id: str) -> int:
    return int(chunk_id[:16], 16)


class QdrantBackend(VectorDBBackend):
    def __init__(self, storage_path: Path, collection_name: str, dimensions: int = 256) -> None:
        ensure_dir(storage_path)
        self.backend_name = "qdrant"
        self.collection_name = collection_name
        self.dimensions = dimensions
        self.client = QdrantClient(path=str(storage_path))
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        exists = False
        if hasattr(self.client, "collection_exists"):
            exists = bool(self.client.collection_exists(collection_name=self.collection_name))
        else:
            try:
                self.client.get_collection(self.collection_name)
                exists = True
            except Exception:  # noqa: BLE001
                exists = False
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qm.VectorParams(size=self.dimensions, distance=qm.Distance.COSINE),
            )

    def upsert_chunks(self, chunks: list[Chunk]) -> list[MutationRecord]:
        if not chunks:
            return []
        points = []
        for chunk in chunks:
            payload = chunk.model_dump(mode="json")
            points.append(
                qm.PointStruct(
                    id=_point_id(chunk.chunk_id),
                    vector=vectorize_text(chunk.content, self.dimensions),
                    payload=payload,
                )
            )
        self.client.upsert(collection_name=self.collection_name, points=points)
        return []

    def delete_chunks(self, chunk_ids: list[str]) -> list[MutationRecord]:
        if chunk_ids:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=qm.PointIdsList(points=[_point_id(chunk_id) for chunk_id in chunk_ids]),
            )
        return []

    def query(self, query: str, top_k: int) -> list[RetrievedChunk]:
        vector = vectorize_text(query, self.dimensions)
        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=vector,
                limit=top_k,
                with_payload=True,
            )
            points = getattr(response, "points", response)
        else:
            points = self.client.search(
                collection_name=self.collection_name,
                query_vector=vector,
                limit=top_k,
                with_payload=True,
            )
        retrieved: list[RetrievedChunk] = []
        for point in points:
            payload = dict(getattr(point, "payload", {}) or {})
            retrieved.append(
                RetrievedChunk(
                    chunk_id=payload["chunk_id"],
                    document_id=payload["document_id"],
                    source_path=payload["source_path"],
                    content=payload["content"],
                    score=float(getattr(point, "score", 0.0)),
                    metadata=payload.get("metadata", {}),
                )
            )
        return retrieved

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        records = self.client.retrieve(collection_name=self.collection_name, ids=[_point_id(chunk_id)], with_payload=True)
        if not records:
            return None
        payload = dict(records[0].payload or {})
        return Chunk.model_validate(payload)

    def dump_state(self) -> VectorDBSnapshot:
        all_chunks: list[Chunk] = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                with_payload=True,
                limit=256,
                offset=offset,
            )
            for point in points:
                payload = dict(point.payload or {})
                all_chunks.append(Chunk.model_validate(payload))
            if offset is None:
                break
        all_chunks.sort(key=lambda item: (item.source_path, item.chunk_index))
        return VectorDBSnapshot(
            backend_name=self.backend_name,
            collection_name=self.collection_name,
            chunk_count=len(all_chunks),
            document_count=len({chunk.document_id for chunk in all_chunks}),
            chunk_ids=[chunk.chunk_id for chunk in all_chunks],
            created_at=utc_timestamp(),
            chunks=all_chunks,
        )
