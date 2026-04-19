from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_config_dir() -> Path:
    for env_var in ("SDT_BENCH_CONFIG_DIR", "CLAB_CONFIG_DIR"):
        configured = os.environ.get(env_var)
        if configured:
            return Path(configured)
    return PROJECT_ROOT / "configs"


def get_benchmark_data_dir() -> Path:
    configured = os.environ.get("SDT_BENCH_DATA_DIR")
    if configured:
        return Path(configured)
    return PROJECT_ROOT / "benchmark_data"
