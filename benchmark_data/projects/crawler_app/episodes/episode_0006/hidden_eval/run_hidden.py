from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    args = parser.parse_args()
    workspace = Path(args.workspace).resolve()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(workspace)
    subprocess.run([sys.executable, "-c", "import crawler_app.settings"], cwd=workspace, env=env, check=True)
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import crawler_app.settings; "
            "assert crawler_app.settings.OFFSITE_MODE == 'allowlist'; assert crawler_app.settings.MIDDLEWARE_STYLE == 'async'",
        ],
        cwd=workspace,
        env=env,
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
