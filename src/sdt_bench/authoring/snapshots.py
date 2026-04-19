from __future__ import annotations

import shutil
from pathlib import Path

from sdt_bench.schemas import RepoSpec, SnapshotManifest
from sdt_bench.utils.fs import ensure_dir, write_json
from sdt_bench.utils.git import checkout_commit, clone_repo
from sdt_bench.utils.subprocess import run_command
from sdt_bench.utils.time import utc_timestamp


def materialize_snapshot(
    *,
    repo_spec: RepoSpec,
    ref: str,
    output_dir: Path,
) -> SnapshotManifest:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    clone_repo(repo_spec.github_url, output_dir)
    checkout_commit(output_dir, ref)
    resolved_commit = run_command(["git", "rev-parse", "HEAD"], cwd=output_dir).stdout.strip()
    manifest = SnapshotManifest(
        repo_name=repo_spec.name,
        source_url=repo_spec.github_url,
        requested_ref=ref,
        resolved_commit=resolved_commit,
        workspace_path=str(output_dir),
        created_at=utc_timestamp(),
    )
    write_json(output_dir / "snapshot_manifest.json", manifest.model_dump(mode="json"))
    return manifest


def default_snapshot_path(root: Path, repo_name: str, ref: str) -> Path:
    slug = ref.replace("/", "_").replace(":", "_")
    return ensure_dir(root / "authoring" / "snapshots" / repo_name) / slug
