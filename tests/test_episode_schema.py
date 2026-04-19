from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from sdt_bench.schemas.episode import EpisodeSpec
from sdt_bench.utils.fs import read_yaml


def test_episode_schema_validates_requests_episode() -> None:
    episode_yaml = Path("benchmark_data/episodes/requests/episode_0001/episode.yaml")
    episode = EpisodeSpec.model_validate(read_yaml(episode_yaml))
    assert episode.repo_name == "requests"
    assert episode.dependency_event.dependency_name == "urllib3"


def test_episode_schema_rejects_empty_visible_docs() -> None:
    payload = read_yaml(Path("benchmark_data/episodes/requests/episode_0001/episode.yaml"))
    payload["visible_doc_paths"] = []
    with pytest.raises(ValidationError):
        EpisodeSpec.model_validate(payload)
