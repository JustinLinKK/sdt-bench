from __future__ import annotations

from pathlib import Path

from clab.schemas.episode import EpisodeSpec


def resolve_visible_doc_paths(episode_dir: Path, episode: EpisodeSpec) -> list[Path]:
    return [episode_dir / relative_path for relative_path in sorted(episode.visible_doc_paths)]
