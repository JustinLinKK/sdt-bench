from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from sdt_bench.schemas.episode import ProgrammingEpisodeSpec
from sdt_bench.schemas.state import TemporalStateSpec
from sdt_bench.schemas.timeline import TimelineSpec
from sdt_bench.utils.fs import read_yaml


def test_programming_episode_schema_validates_toy_episode() -> None:
    episode_yaml = Path("benchmark_data/projects/toy/episodes/episode_0001/episode.yaml")
    episode = ProgrammingEpisodeSpec.model_validate(read_yaml(episode_yaml))
    assert episode.timeline_id == "toy"
    assert episode.event_id == "toy_event_0001"
    assert episode.project_id == "toy"


def test_timeline_schema_requires_ordered_lists() -> None:
    payload = read_yaml(Path("benchmark_data/projects/toy/timeline.yaml"))
    payload["episode_ids"] = []
    with pytest.raises(ValidationError):
        TimelineSpec.model_validate(payload)


def test_temporal_state_schema_requires_install_command() -> None:
    payload = read_yaml(Path("benchmark_data/projects/toy/states/toy_2026_01/state.yaml"))
    payload["environment"]["install_command"] = ""
    with pytest.raises(ValidationError):
        TemporalStateSpec.model_validate(payload)
