from __future__ import annotations

import sys
from pathlib import Path

from typer.testing import CliRunner

from sdt_bench.cli import app
from sdt_bench.utils.fs import read_json, read_jsonl
from tests.helpers import make_temp_config, toy_episode_path


def test_external_agent_reads_input_bundle_and_writes_required_output(tmp_path: Path, monkeypatch) -> None:
    config_dir = make_temp_config(tmp_path)
    monkeypatch.setenv("SDT_BENCH_CONFIG_DIR", str(config_dir))

    script_path = tmp_path / "agent.py"
    script_path.write_text(
        """
import json
import sys
from pathlib import Path

input_dir = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
manifest = json.loads((input_dir / "manifest.json").read_text())
memory_manifest = json.loads((input_dir / "memory" / "manifest.json").read_text())
assert not (input_dir / "hidden_eval").exists()
docs_manifest = json.loads((input_dir / "docs" / "manifest.json").read_text())
output_dir.mkdir(parents=True, exist_ok=True)
(output_dir / "plan.json").write_text(json.dumps({"summary": "external ok", "steps": ["inspect bundle"]}))
(output_dir / "ingestion_decision.json").write_text(json.dumps({
    "strategy": "selected_visible",
    "ingest_visible_docs": True,
    "selected_visible_doc_paths": [docs_manifest["documents"][0]["path"]],
    "acquisitions": [docs_manifest["documents"][0]["path"]],
    "reason": "ingest first visible doc"
}))
(output_dir / "retrieval_decision.json").write_text(json.dumps({"query": "toy greet", "top_k": 1, "reason": "smoke"}))
(output_dir / "retrieval_trace.json").write_text(json.dumps({
    "episode_id": manifest["episode_id"],
    "query": "toy greet",
    "retrieved_chunk_ids": [],
    "retrieved_document_ids": [],
    "scores": [],
    "freshness_labels": [],
    "timestamp": "2026-01-01T00:00:00+00:00"
}))
(output_dir / "patch.diff").write_text("")
(output_dir / "citations.json").write_text(json.dumps({"citations": [docs_manifest["documents"][0]["path"]]}))
(output_dir / "memory_mutations.jsonl").write_text("")
(output_dir / "review.json").write_text(json.dumps({"summary": f"memory={memory_manifest['chunk_count']}", "concerns": []}))
""".strip(),
        encoding="utf-8",
    )

    runner = CliRunner()
    episode_dir = toy_episode_path("episode_0001")
    result = runner.invoke(app, ["materialize-step", str(episode_dir), "--agent", "external"])
    assert result.exit_code == 0, result.stdout
    result = runner.invoke(
        app,
        [
            "run-step",
            str(episode_dir),
            "--agent",
            "external",
            "--agent-command",
            f'"{sys.executable}" "{script_path}" "{{input_dir}}" "{{output_dir}}"',
        ],
    )
    assert result.exit_code == 0, result.stdout

    runs_root = tmp_path / "runtime" / "runs" / "toy" / "external"
    run_id = (runs_root / "last_run.txt").read_text(encoding="utf-8").strip()
    step_root = runs_root / run_id / "steps" / "000__episode_0001"
    output_dir = step_root / "output"
    assert (output_dir / "plan.json").exists()
    assert (output_dir / "review.json").exists()
    review = read_json(output_dir / "review.json")
    assert review["summary"] == "memory=0"
    assert read_jsonl(output_dir / "memory_mutations.jsonl") == []
