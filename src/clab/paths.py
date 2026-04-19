from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_config_dir() -> Path:
    configured = os.environ.get("CLAB_CONFIG_DIR")
    if configured:
        return Path(configured)
    return PROJECT_ROOT / "configs"
