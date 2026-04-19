from __future__ import annotations

from pathlib import Path

from clab.benchmark.loader import hash_visible_docs, load_episode_spec, validate_episode
from clab.schemas.repo import RepoSpec


def test_loader_reads_requests_episode() -> None:
    episode_dir, episode = load_episode_spec(Path("data/episodes/requests/episode_0001"))
    assert episode_dir.name == "episode_0001"
    hashes = hash_visible_docs(episode_dir, episode)
    assert set(hashes) == set(episode.visible_doc_paths)


def test_validate_episode_returns_summary() -> None:
    episode_dir, episode = load_episode_spec(Path("data/episodes/requests/episode_0001"))
    repo_spec = RepoSpec.model_validate(
        {
            "name": "requests",
            "github_url": "https://github.com/psf/requests.git",
            "default_branch": "main",
            "language": "python",
            "package_manager": "pip",
            "install_command": "python -m pip install -e .",
            "test_command": "python -m pytest",
            "dockerfile_path": "Dockerfile",
            "supported_dependency_files": ["pyproject.toml"],
            "notes": "test",
        }
    )
    summary = validate_episode(episode_dir, episode, repo_spec)
    assert summary["repo_name"] == "requests"
