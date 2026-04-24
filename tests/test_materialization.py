from __future__ import annotations

from pathlib import Path

from sdt_bench.benchmark import load_global_config, load_step_bundle, materialize_step
from sdt_bench.env import create_timeline_run_layout, set_last_run
from tests.helpers import make_temp_config, toy_episode_path


def test_materialize_step_copies_product_snapshot_and_state_tests(tmp_path: Path, monkeypatch) -> None:
    config_dir = make_temp_config(tmp_path)
    monkeypatch.setenv("SDT_BENCH_CONFIG_DIR", str(config_dir))

    bundle = load_step_bundle(toy_episode_path("episode_0001"))
    global_config = load_global_config(config_dir)
    timeline_layout = create_timeline_run_layout(
        global_config,
        timeline_id=bundle.timeline.timeline_id,
        agent_name="inspect",
        run_id="copy_check",
    )
    set_last_run(timeline_layout)

    result = materialize_step(
        global_config=global_config,
        bundle=bundle,
        project_spec=bundle.project,
        timeline_layout=timeline_layout,
        step_index=0,
        agent_name="inspect",
        memory_mode="persistent",
        memory_chunks=[],
    )

    workspace = Path(result["workspace"])
    snapshot_file = bundle.from_state_dir / bundle.from_state.snapshot_root / "toy_pkg" / "__init__.py"
    assert (workspace / "toy_pkg" / "__init__.py").read_text(encoding="utf-8") == snapshot_file.read_text(encoding="utf-8")
    assert (workspace / "tests" / "visible" / "test_smoke.py").exists()
