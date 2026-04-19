from __future__ import annotations

from pathlib import Path
from typing import Any

from clab.paths import get_config_dir
from clab.schemas.episode import EpisodeSpec
from clab.schemas.repo import RepoSpec
from clab.utils.fs import read_yaml
from clab.utils.hashing import sha256_file


def resolve_episode_dir(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_file():
        return candidate.parent.resolve()
    return candidate.resolve()


def load_global_config(config_dir: Path | None = None) -> dict[str, Any]:
    selected = config_dir or get_config_dir()
    return read_yaml(selected / "global.yaml")


def load_repo_spec(repo_name: str, config_dir: Path | None = None) -> RepoSpec:
    selected = config_dir or get_config_dir()
    return RepoSpec.model_validate(read_yaml(selected / "repos" / f"{repo_name}.yaml"))


def load_episode_spec(path: str | Path) -> tuple[Path, EpisodeSpec]:
    episode_dir = resolve_episode_dir(path)
    return episode_dir, EpisodeSpec.model_validate(read_yaml(episode_dir / "episode.yaml"))


def hash_visible_docs(episode_dir: Path, episode: EpisodeSpec) -> dict[str, str]:
    return {
        visible_path: sha256_file(episode_dir / visible_path)
        for visible_path in sorted(episode.visible_doc_paths)
    }


def validate_episode(
    episode_dir: Path, episode: EpisodeSpec, repo_spec: RepoSpec
) -> dict[str, Any]:
    missing_paths: list[str] = []
    for relative_path in episode.visible_doc_paths:
        if not (episode_dir / relative_path).exists():
            missing_paths.append(relative_path)
    hidden_manifest = episode_dir / episode.hidden_test_manifest
    if not hidden_manifest.exists():
        missing_paths.append(episode.hidden_test_manifest)
    for relative_path in episode.dependency_event.gold_mutation_paths:
        if not (episode_dir / relative_path).exists():
            missing_paths.append(relative_path)
    if missing_paths:
        raise FileNotFoundError(f"Missing episode files: {', '.join(missing_paths)}")

    if not repo_spec.install_command.strip():
        raise ValueError("RepoSpec.install_command must be defined")
    if not episode.hidden_test_command.strip():
        raise ValueError("Episode hidden_test_command must be defined")

    return {
        "episode_id": episode.episode_id,
        "repo_name": episode.repo_name,
        "visible_doc_hashes": hash_visible_docs(episode_dir, episode),
        "hidden_test_manifest": str(hidden_manifest),
        "expected_files_of_interest": episode.expected_files_of_interest,
    }
