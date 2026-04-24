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
    subprocess.run([sys.executable, "-c", "import workflow_app.config"], cwd=workspace, env=env, check=True)
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import workflow_app.config; "
            "assert workflow_app.config.INTEGRATION_BACKEND == 'deployment'; assert workflow_app.config.TASK_STYLE == 'async_compatible'",
        ],
        cwd=workspace,
        env=env,
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
