from __future__ import annotations

from pathlib import Path


def docker_build_recipe(dockerfile_path: str | None, project_root: Path) -> dict[str, str | None]:
    return {
        "dockerfile_path": str(project_root / dockerfile_path) if dockerfile_path else None,
        "image_reference": None,
    }
