from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_check(script: str, workspace: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(workspace)
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=workspace,
        check=True,
        env=env,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    args = parser.parse_args()
    workspace = Path(args.workspace).resolve()

    subprocess.run(["uv", "pip", "install", "urllib3==2.0.7"], cwd=workspace, check=True)

    run_check("import requests", workspace)
    run_check(
        "import requests; r = requests.Request('GET', 'https://example.com');"
        " p = r.prepare(); assert p.method == 'GET'; assert p.url.startswith('https://')",
        workspace,
    )
    run_check("from requests.adapters import HTTPAdapter; HTTPAdapter()", workspace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
