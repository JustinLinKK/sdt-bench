"""Pull package versions, changelogs, advisories, and dependency metadata.

Outputs data/releases/<package>.jsonl with one row per version:
  {version, published_at, is_default, license_codes, advisories[], changelog_url}
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from common import DATA, load_config, write_jsonl

DEPS_DEV = "https://api.deps.dev/v3"
OSV = "https://api.osv.dev/v1/query"
UA = {"User-Agent": "llm-os-benchmark-pilot/0.1"}


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=20))
def _get(url: str) -> dict:
    r = requests.get(url, headers=UA, timeout=30)
    r.raise_for_status()
    return r.json()


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=20))
def _post(url: str, payload: dict) -> dict:
    r = requests.post(url, json=payload, headers=UA, timeout=30)
    r.raise_for_status()
    return r.json()


def list_versions(ecosystem: str, package: str) -> list[dict]:
    url = f"{DEPS_DEV}/systems/{ecosystem}/packages/{package}"
    data = _get(url)
    return data.get("versions", [])


def osv_advisories(ecosystem: str, package: str, version: str) -> list[dict]:
    payload = {"package": {"name": package, "ecosystem": ecosystem}, "version": version}
    try:
        return _post(OSV, payload).get("vulns", []) or []
    except requests.HTTPError:
        return []


def harvest_repo(repo_cfg: dict, max_versions: int = 80) -> Path:
    pkg = repo_cfg["package"]
    ecosystem = repo_cfg["ecosystem"]
    out = DATA / "releases" / f"{pkg}.jsonl"
    rows: list[dict] = []

    versions = list_versions(ecosystem, pkg)
    # deps.dev returns oldest first; take tail for recency.
    versions = versions[-max_versions:]

    for v in versions:
        ver_key = v.get("versionKey", {})
        version = ver_key.get("version")
        if not version:
            continue
        published = v.get("publishedAt") or v.get("publishDate")
        adv = osv_advisories(ecosystem, pkg, version)
        rows.append(
            {
                "package": pkg,
                "ecosystem": ecosystem,
                "version": version,
                "published_at": published,
                "is_default": v.get("isDefault", False),
                "license_codes": v.get("licenses", {}).get("spdxIds", []) if isinstance(v.get("licenses"), dict) else [],
                "advisories": [a.get("id") for a in adv],
                "changelog_url": f"https://github.com/{repo_cfg['github']}/releases/tag/{version}",
            }
        )
        time.sleep(0.15)  # gentle to OSV

    write_jsonl(out, rows)
    print(f"  wrote {len(rows)} rows -> {out.relative_to(DATA.parent)}")
    return out


def main() -> int:
    cfg = load_config()
    for repo in cfg["repos"]:
        print(f"[harvest] {repo['package']}")
        try:
            harvest_repo(repo)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! failed: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
