"""Chunk docs + selected code into stable-ID chunks and upsert into Qdrant.

Chunk layout
------------
  doc_chunks  : ~900-char windows over rst/md/txt/changelog with 150 overlap
  code_chunks : one chunk per top-level def/class in .py files (regex-driven)

Each chunk row:
  {chunk_id, package, version, path, kind, content}

Chunk ID = common.chunk_id(path, content) — stable across re-runs provided
(path, content) bytes are unchanged. This is what makes update/delete
auditable later.

Vector store
------------
Local Qdrant in-process (qdrant_client.QdrantClient(path=...)). Collection
per package so diffs between versions are contained.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterator

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from common import DATA, OpenRouterClient, chunk_id, load_config, write_jsonl

DOC_EXTS = {".md", ".rst", ".txt"}
DOC_CHUNK = 900
DOC_OVERLAP = 150

TOP_DEF = re.compile(r"^(def |class |async def )", re.MULTILINE)


def _doc_windows(text: str) -> list[str]:
    if not text.strip():
        return []
    out: list[str] = []
    i = 0
    while i < len(text):
        out.append(text[i : i + DOC_CHUNK])
        i += DOC_CHUNK - DOC_OVERLAP
    return out


def _code_blocks(text: str) -> list[str]:
    """Split on top-level def/class lines; keep signature + body window."""
    spans: list[str] = []
    starts = [m.start() for m in TOP_DEF.finditer(text)]
    if not starts:
        return [text[:DOC_CHUNK]] if text.strip() else []
    starts.append(len(text))
    for a, b in zip(starts, starts[1:]):
        block = text[a:b]
        if len(block) > 1800:
            block = block[:1800]
        if block.strip():
            spans.append(block)
    return spans


def iter_chunks(snap_dir: Path, manifest: dict) -> Iterator[dict]:
    repo = Path(manifest["repo_path"])
    for rel in manifest["doc_files"]:
        p = repo / rel
        if p.suffix.lower() not in DOC_EXTS and not p.name.startswith(("CHANGES", "CHANGELOG", "HISTORY", "README")):
            continue
        try:
            text = p.read_text(errors="ignore")
        except Exception:
            continue
        for chunk in _doc_windows(text):
            yield {
                "path": rel,
                "kind": "doc",
                "content": chunk,
            }
    for rel in manifest["code_files"]:
        p = repo / rel
        try:
            text = p.read_text(errors="ignore")
        except Exception:
            continue
        for chunk in _code_blocks(text):
            yield {"path": rel, "kind": "code", "content": chunk}


def _checkout_for_snapshot(manifest: dict) -> None:
    """Make the shared clone match the commit this snapshot was taken at.

    This is critical: materialise_snapshot left the repo at the LAST tag
    it processed, so naive reads of repo_path here would compute chunks
    for that final version against every snapshot directory.
    """
    repo = Path(manifest["repo_path"])
    subprocess.run(
        ["git", "checkout", "--quiet", "--force", manifest["commit"]],
        cwd=repo, check=True, capture_output=True,
    )


def build_manifest(snap_dir: Path) -> Path:
    manifest = json.loads((snap_dir / "manifest.json").read_text())
    _checkout_for_snapshot(manifest)
    rows: list[dict] = []
    for c in iter_chunks(snap_dir, manifest):
        cid = chunk_id(c["path"], c["content"])
        rows.append({
            "chunk_id": cid,
            "package": manifest["package"],
            "version": manifest["to_version"],
            "path": c["path"],
            "kind": c["kind"],
            "content": c["content"],
        })
    out = snap_dir / "chunks.jsonl"
    write_jsonl(out, rows)
    print(f"[chunk] {manifest['package']}@{manifest['to_version']}: {len(rows)} chunks")
    return out


def embed_and_upsert(snap_dir: Path, collection_dir: Path, model: str, dry_run: bool) -> None:
    chunks_path = snap_dir / "chunks.jsonl"
    rows = [json.loads(line) for line in chunks_path.read_text().splitlines() if line.strip()]
    if not rows:
        return
    collection_dir.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(collection_dir))
    pkg = rows[0]["package"]
    ver = rows[0]["version"]

    if dry_run:
        print(f"[qdrant:dry] {pkg}@{ver}: would upsert {len(rows)} points")
        return

    try:
        client.get_collection(pkg)
    except Exception:
        client.create_collection(
            collection_name=pkg,
            vectors_config=qm.VectorParams(size=1536, distance=qm.Distance.COSINE),
        )

    or_client = OpenRouterClient.from_env()
    batch = 32
    points: list[qm.PointStruct] = []
    for i in range(0, len(rows), batch):
        window = rows[i : i + batch]
        vectors = or_client.embed(model, [w["content"] for w in window])
        for w, v in zip(window, vectors):
            # Qdrant point IDs must be uint64 or UUID; reuse chunk_id hex as UUID-prefix.
            pid = int(w["chunk_id"], 16) & ((1 << 63) - 1)
            points.append(qm.PointStruct(
                id=pid,
                vector=v,
                payload={k: w[k] for k in ("chunk_id", "package", "version", "path", "kind")},
            ))
    client.upsert(collection_name=pkg, points=points)
    print(f"[qdrant] {pkg}@{ver}: upserted {len(points)} points")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Skip embedding/upsert; only write chunks.jsonl")
    args = ap.parse_args()

    cfg = load_config()
    model = cfg["pilot"]["embedding_model"]
    coll_dir = DATA / "vector_db"

    snap_root = DATA / "snapshots"
    for pkg_dir in sorted(p for p in snap_root.iterdir() if p.is_dir()):
        for ver_dir in sorted(v for v in pkg_dir.iterdir() if v.is_dir()):
            if not (ver_dir / "manifest.json").exists():
                continue
            build_manifest(ver_dir)
            try:
                embed_and_upsert(ver_dir, coll_dir, model, args.dry_run)
            except Exception as exc:  # noqa: BLE001
                print(f"  ! embed/upsert failed for {ver_dir}: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
