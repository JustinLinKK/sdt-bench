from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sdt_bench.benchmark.loader import LoadedStep, load_state_docs_manifest
from sdt_bench.schemas import VisibleDocumentSpec


@dataclass(slots=True)
class VisibleDocEntry:
    source_path: Path
    relative_path: str
    available_at: str
    source_kind: str


def resolve_visible_docs(bundle: LoadedStep) -> list[VisibleDocEntry]:
    docs: list[VisibleDocEntry] = []
    state_manifest = load_state_docs_manifest(bundle.to_state_dir, bundle.to_state)
    for document in state_manifest.documents:
        if document.available_at <= bundle.to_state.timestamp:
            docs.append(
                VisibleDocEntry(
                    source_path=bundle.to_state_dir / document.path,
                    relative_path=document.path.replace("\\", "/"),
                    available_at=document.available_at,
                    source_kind=document.source or "state",
                )
            )

    event_time = bundle.event.available_at
    if event_time <= bundle.to_state.timestamp:
        for relative_path in bundle.event.visible_doc_paths:
            normalized = relative_path.replace("\\", "/")
            docs.append(
                VisibleDocEntry(
                    source_path=bundle.event_dir / relative_path,
                    relative_path=normalized,
                    available_at=event_time,
                    source_kind="event",
                )
            )

    deduped: dict[str, VisibleDocEntry] = {}
    for item in docs:
        deduped[item.relative_path] = item
    return [deduped[path] for path in sorted(deduped)]


def build_docs_manifest(entries: list[VisibleDocEntry]) -> list[VisibleDocumentSpec]:
    return [
        VisibleDocumentSpec(
            path=entry.relative_path,
            available_at=entry.available_at,
            source=entry.source_kind,
        )
        for entry in entries
    ]
