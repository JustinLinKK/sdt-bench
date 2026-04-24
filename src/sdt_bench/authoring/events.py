from __future__ import annotations

from pathlib import Path

from packaging.version import InvalidVersion, Version

from sdt_bench.schemas import EventStreamRecord, ProjectSpec, ReleaseRecord
from sdt_bench.utils.fs import read_jsonl, write_jsonl
from sdt_bench.utils.hashing import sha256_text


def classify_event_type(old_version: str, new_version: str, new_advisories: list[str]) -> str:
    if new_advisories:
        return "security"
    try:
        old = Version(old_version)
        new = Version(new_version)
    except InvalidVersion:
        return "synthetic"
    old_release = old.release
    new_release = new.release
    if len(old_release) >= 1 and len(new_release) >= 1 and old_release[0] != new_release[0]:
        return "major"
    if len(old_release) >= 2 and len(new_release) >= 2 and old_release[1] != new_release[1]:
        return "minor"
    if len(old_release) >= 3 and len(new_release) >= 3 and old_release[2] != new_release[2]:
        return "patch"
    return "synthetic"


def build_event_stream(
    project_spec: ProjectSpec,
    releases: list[ReleaseRecord],
    *,
    max_events: int | None = None,
) -> list[EventStreamRecord]:
    ordered = sorted(releases, key=lambda item: Version(item.version))
    events: list[EventStreamRecord] = []
    for previous, current in zip(ordered, ordered[1:], strict=False):
        new_advisories = sorted(set(current.advisories) - set(previous.advisories))
        event_type = classify_event_type(previous.version, current.version, new_advisories)
        event_id = sha256_text(f"{project_spec.project_id}:{project_spec.framework_package}:{previous.version}:{current.version}")[:16]
        events.append(
            EventStreamRecord(
                event_id=event_id,
                project_id=project_spec.project_id,
                dependency_name=project_spec.framework_package,
                ecosystem="pypi",
                old_version=previous.version,
                new_version=current.version,
                event_type=event_type,
                published_at=current.published_at,
                new_advisories=new_advisories,
                metadata={"source": "release_stream"},
            )
        )
    return events[:max_events] if max_events is not None else events


def read_release_records(path: Path) -> list[ReleaseRecord]:
    return [ReleaseRecord.model_validate(item) for item in read_jsonl(path)]


def default_event_output_path(root: Path, project_id: str) -> Path:
    return root / "authoring" / "events" / f"{project_id}.jsonl"


def write_event_stream(path: Path, events: list[EventStreamRecord]) -> None:
    write_jsonl(path, [event.model_dump(mode="json") for event in events])
