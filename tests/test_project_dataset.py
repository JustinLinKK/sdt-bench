from __future__ import annotations

from sdt_bench.benchmark import load_project_spec, load_step_bundle, load_timeline_spec, validate_step
from sdt_bench.paths import PROJECT_ROOT
from sdt_bench.utils.fs import read_yaml


def test_manifest_lists_project_first_dataset() -> None:
    manifest = read_yaml(PROJECT_ROOT / "benchmark_data" / "manifest.yaml")
    assert manifest["project_ids"] == ["toy", "crawler_app", "workflow_app", "rag_app"]


def test_all_projects_validate_every_episode() -> None:
    manifest = read_yaml(PROJECT_ROOT / "benchmark_data" / "manifest.yaml")
    total_episodes = 0

    for project_id in manifest["project_ids"]:
        project_dir, project = load_project_spec(project_id)
        _, timeline = load_timeline_spec(project_dir / "timeline.yaml")
        assert project.project_id == project_id
        assert timeline.timeline_id == project_id
        assert timeline.project_id == project_id

        for episode_id in timeline.episode_ids:
            bundle = load_step_bundle(project_dir / "episodes" / episode_id)
            summary = validate_step(bundle)
            assert summary["project_id"] == project_id
            total_episodes += 1

    assert total_episodes == 26
