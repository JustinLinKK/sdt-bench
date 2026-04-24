from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import httpx
from packaging.version import InvalidVersion, Version

from sdt_bench.schemas import ProjectSpec, ReleaseRecord
from sdt_bench.utils.fs import write_jsonl

PYPI_URL = "https://pypi.org/pypi/{package_name}/json"
OSV_URL = "https://api.osv.dev/v1/query"


def harvest_release_records(
    project_spec: ProjectSpec,
    *,
    max_versions: int = 50,
    include_advisories: bool = True,
    timeout_seconds: int = 30,
) -> list[ReleaseRecord]:
    package_name = project_spec.framework_package
    ecosystem = "PYPI"
    with httpx.Client(timeout=timeout_seconds, follow_redirects=True) as client:
        response = client.get(PYPI_URL.format(package_name=package_name))
        response.raise_for_status()
        payload = response.json()

        releases: list[tuple[Version, str, str | None]] = []
        for version_text, files in payload.get("releases", {}).items():
            try:
                parsed = Version(version_text)
            except InvalidVersion:
                continue
            published_at = None
            if files:
                timestamps = [item.get("upload_time_iso_8601") for item in files if item.get("upload_time_iso_8601")]
                published_at = min(timestamps) if timestamps else None
            releases.append((parsed, version_text, published_at))

        releases.sort(key=lambda item: item[0])
        selected = releases[-max_versions:]
        records = [
            ReleaseRecord(
                package_name=package_name,
                ecosystem=ecosystem,
                version=version_text,
                published_at=published_at,
                advisories=_fetch_osv_advisories(
                    client=client,
                    package_name=package_name,
                    ecosystem=ecosystem,
                    version=version_text,
                )
                if include_advisories
                else [],
                metadata={"project_id": project_spec.project_id},
            )
            for _, version_text, published_at in selected
        ]
    return records


def _fetch_osv_advisories(
    *,
    client: httpx.Client,
    package_name: str,
    ecosystem: str,
    version: str,
) -> list[str]:
    payload = {
        "version": version,
        "package": {"name": package_name, "ecosystem": ecosystem},
    }
    response = client.post(OSV_URL, json=payload)
    response.raise_for_status()
    body = response.json()
    return sorted(item.get("id", "") for item in body.get("vulns", []) if item.get("id"))


def default_release_output_path(root: Path, project_id: str) -> Path:
    return root / "authoring" / "releases" / f"{project_id}.jsonl"


def write_release_records(path: Path, records: list[ReleaseRecord]) -> None:
    write_jsonl(path, [record.model_dump(mode="json") for record in records])


def timestamp_now() -> str:
    return datetime.now(UTC).isoformat()
