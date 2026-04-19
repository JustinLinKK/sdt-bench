from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    args = parser.parse_args()
    workspace = Path(args.workspace).resolve()
    assert (workspace / "requests" / "__init__.py").exists()
    assert (workspace / "requests" / "adapters.py").exists()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
