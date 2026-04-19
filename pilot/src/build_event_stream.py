"""Convert release histories into a timed upgrade-event stream.

Event bucket rules (SemVer via packaging.version):
  patch    = micro bumped, major/minor unchanged
  minor    = minor bumped, major unchanged
  major    = major bumped
  security = any bump where the 'to' version has a new OSV advisory not
             present in the 'from' version (overrides bucket above).

Output: data/events/<package>.jsonl, ordered by published_at.
One row per consecutive-version transition with max_events_per_repo cap.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from packaging.version import InvalidVersion, Version

from common import DATA, load_config, read_jsonl, write_jsonl


def _parse(v: str) -> Version | None:
    try:
        return Version(v)
    except InvalidVersion:
        return None


def _bucket(a: Version, b: Version) -> str:
    if b.release[0] > a.release[0]:
        return "major"
    if len(b.release) > 1 and b.release[1] > a.release[1]:
        return "minor"
    return "patch"


def _parse_date(raw: str | None) -> datetime | None:
    if not raw:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def build_for_package(pkg: str, max_events: int) -> Path:
    rel = list(read_jsonl(DATA / "releases" / f"{pkg}.jsonl"))
    rows: list[dict] = []
    for r in rel:
        v = _parse(r["version"])
        d = _parse_date(r.get("published_at"))
        if v and not v.is_prerelease and d:
            rows.append({**r, "_v": v, "_d": d})
    rows.sort(key=lambda r: (r["_d"], r["_v"]))

    events: list[dict] = []
    for prev, cur in zip(rows, rows[1:]):
        bucket = _bucket(prev["_v"], cur["_v"])
        new_adv = sorted(set(cur.get("advisories", [])) - set(prev.get("advisories", [])))
        if new_adv:
            bucket = "security"
        events.append(
            {
                "package": pkg,
                "from_version": prev["version"],
                "to_version": cur["version"],
                "event_time": cur["_d"].isoformat(),
                "bucket": bucket,
                "new_advisories": new_adv,
                "changelog_url": cur.get("changelog_url"),
            }
        )

    # Keep last N events (most recent) to bound pilot cost.
    events = events[-max_events:]
    out = DATA / "events" / f"{pkg}.jsonl"
    write_jsonl(out, events)
    print(f"[events] {pkg}: {len(events)} events -> {out.relative_to(DATA.parent)}")
    return out


def main() -> int:
    cfg = load_config()
    cap = cfg["pilot"]["max_events_per_repo"]
    for repo in cfg["repos"]:
        build_for_package(repo["package"], cap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
