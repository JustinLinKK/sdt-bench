from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sdt_bench.paths import get_benchmark_data_dir, get_config_dir
from sdt_bench.schemas import (
    DependencyEventSpec,
    DocumentManifest,
    ProgrammingEpisodeSpec,
    RepoSpec,
    TemporalStateSpec,
    TimelineSpec,
)
from sdt_bench.utils.fs import read_yaml


@dataclass(slots=True)
class LoadedStep:
    episode_dir: Path
    event_dir: Path
    from_state_dir: Path
    to_state_dir: Path
    timeline_path: Path
    episode: ProgrammingEpisodeSpec
    event: DependencyEventSpec
    from_state: TemporalStateSpec
    to_state: TemporalStateSpec
    timeline: TimelineSpec


def resolve_step_dir(path: str | Path) -> Path:
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


def load_timeline_spec(path: str | Path) -> tuple[Path, TimelineSpec]:
    timeline_path = Path(path).resolve()
    return timeline_path, TimelineSpec.model_validate(read_yaml(timeline_path))


def load_state_spec(path: str | Path) -> tuple[Path, TemporalStateSpec]:
    state_dir = resolve_step_dir(path)
    return state_dir, TemporalStateSpec.model_validate(read_yaml(state_dir / "state.yaml"))


def load_event_spec(path: str | Path) -> tuple[Path, DependencyEventSpec]:
    event_dir = resolve_step_dir(path)
    return event_dir, DependencyEventSpec.model_validate(read_yaml(event_dir / "event.yaml"))


def load_episode_spec(path: str | Path) -> tuple[Path, ProgrammingEpisodeSpec]:
    episode_dir = resolve_step_dir(path)
    return episode_dir, ProgrammingEpisodeSpec.model_validate(read_yaml(episode_dir / "episode.yaml"))


def load_step_bundle(path: str | Path, benchmark_data_dir: Path | None = None) -> LoadedStep:
    episode_dir, episode = load_episode_spec(path)
    data_dir = benchmark_data_dir or get_benchmark_data_dir()
    timeline_path = data_dir / "timelines" / f"{episode.repo_name}.yaml"
    _, timeline = load_timeline_spec(timeline_path)
    event_dir = data_dir / "events" / episode.repo_name / episode.event_id
    from_state_dir = data_dir / "states" / episode.repo_name / episode.from_state_id
    to_state_dir = data_dir / "states" / episode.repo_name / episode.to_state_id
    _, event = load_event_spec(event_dir)
    _, from_state = load_state_spec(from_state_dir)
    _, to_state = load_state_spec(to_state_dir)
    return LoadedStep(
        episode_dir=episode_dir,
        event_dir=event_dir,
        from_state_dir=from_state_dir,
        to_state_dir=to_state_dir,
        timeline_path=timeline_path,
        episode=episode,
        event=event,
        from_state=from_state,
        to_state=to_state,
        timeline=timeline,
    )


def load_state_docs_manifest(
    state_dir: Path,
    state: TemporalStateSpec,
) -> DocumentManifest:
    manifest_path = state_dir / state.docs_manifest_path
    if not manifest_path.exists():
        return DocumentManifest(documents=[])
    payload = read_yaml(manifest_path) or {}
    return DocumentManifest.model_validate(payload)


def validate_step(bundle: LoadedStep, repo_spec: RepoSpec) -> dict[str, Any]:
    missing_paths: list[str] = []

    if bundle.timeline.timeline_id != bundle.episode.timeline_id:
        raise ValueError("Episode timeline_id does not match the loaded timeline")
    if bundle.episode.repo_name != bundle.timeline.repo_name:
        raise ValueError("Episode repo_name does not match the loaded timeline")
    if bundle.episode.event_id != bundle.event.event_id:
        raise ValueError("Episode event_id does not match the loaded event")
    if bundle.event.from_state_id != bundle.episode.from_state_id:
        raise ValueError("Event from_state_id does not match the episode")
    if bundle.event.to_state_id != bundle.episode.to_state_id:
        raise ValueError("Event to_state_id does not match the episode")
    if bundle.from_state.state_id != bundle.episode.from_state_id:
        raise ValueError("Loaded from_state does not match the episode")
    if bundle.to_state.state_id != bundle.episode.to_state_id:
        raise ValueError("Loaded to_state does not match the episode")
    if bundle.episode.episode_id not in bundle.timeline.episode_ids:
        raise ValueError("Episode is not listed in the timeline")
    if bundle.from_state.state_id not in bundle.timeline.state_ids:
        raise ValueError("from_state is not listed in the timeline")
    if bundle.to_state.state_id not in bundle.timeline.state_ids:
        raise ValueError("to_state is not listed in the timeline")

    visible_failure = bundle.episode_dir / bundle.episode.visible_failure_path
    if not visible_failure.exists():
        missing_paths.append(str(visible_failure))
    hidden_manifest = bundle.episode_dir / bundle.episode.hidden_test_manifest
    if not hidden_manifest.exists():
        missing_paths.append(str(hidden_manifest))

    event_required_paths = [
        *bundle.event.visible_doc_paths,
        *bundle.event.gold_mutation_paths,
    ]
    if bundle.event.expected_retrieval_path:
        event_required_paths.append(bundle.event.expected_retrieval_path)
    for relative_path in event_required_paths:
        if not (bundle.event_dir / relative_path).exists():
            missing_paths.append(str(bundle.event_dir / relative_path))

    manifest = load_state_docs_manifest(bundle.to_state_dir, bundle.to_state)
    for document in manifest.documents:
        if not (bundle.to_state_dir / document.path).exists():
            missing_paths.append(str(bundle.to_state_dir / document.path))

    if missing_paths:
        raise FileNotFoundError(f"Missing step files: {', '.join(missing_paths)}")
    if not repo_spec.install_command.strip():
        raise ValueError("RepoSpec.install_command must be defined")
    if not bundle.to_state.environment.install_command.strip():
        raise ValueError("State environment install_command must be defined")
    if not bundle.episode.hidden_test_command.strip():
        raise ValueError("Episode hidden_test_command must be defined")

    return {
        "timeline_id": bundle.timeline.timeline_id,
        "episode_id": bundle.episode.episode_id,
        "repo_name": bundle.episode.repo_name,
        "from_state_id": bundle.from_state.state_id,
        "to_state_id": bundle.to_state.state_id,
        "event_id": bundle.event.event_id,
        "visible_docs": len(bundle.event.visible_doc_paths) + len(manifest.documents),
        "hidden_test_manifest": str(hidden_manifest),
    }
