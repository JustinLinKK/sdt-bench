from __future__ import annotations

import sys
from pathlib import Path

from typer.testing import CliRunner

from sdt_bench.cli import app
from sdt_bench.utils.fs import read_json
from tests.helpers import make_temp_config, toy_episode_path, toy_timeline_path


def test_timeline_smoke_and_memory_modes(tmp_path: Path, monkeypatch) -> None:
    config_dir = make_temp_config(tmp_path)
    monkeypatch.setenv("SDT_BENCH_CONFIG_DIR", str(config_dir))
    runner = CliRunner()

    result = runner.invoke(app, ["validate-step", str(toy_episode_path("episode_0001"))])
    assert result.exit_code == 0, result.stdout

    result = runner.invoke(
        app,
        [
            "materialize-step",
            str(toy_episode_path("episode_0001")),
            "--agent",
            "inspect",
            "--run-id",
            "inspect",
        ],
    )
    assert result.exit_code == 0, result.stdout
    inspect_root = tmp_path / "runtime" / "runs" / "toy" / "inspect" / "inspect" / "steps" / "000__episode_0001"
    docs_manifest = read_json(inspect_root / "input" / "docs" / "manifest.json")
    assert "docs/deprecation_notice.md" not in [item["path"] for item in docs_manifest["documents"]]
    environment = read_json(inspect_root / "harness" / "environment.json")
    assert environment["install"]["offline_mode"] is True
    assert environment["install"]["environment_overrides"]["PIP_NO_INDEX"] == "1"
    assert not (inspect_root / "input" / "hidden_eval").exists()

    result = runner.invoke(
        app,
        [
            "run-timeline",
            str(toy_timeline_path()),
            "--agent",
            "baseline:dummy",
        ],
    )
    assert result.exit_code == 0, result.stdout

    runs_root = tmp_path / "runtime" / "runs" / "toy" / "baseline__dummy"
    run_id = (runs_root / "last_run.txt").read_text(encoding="utf-8").strip()
    timeline_result = read_json(runs_root / run_id / "timeline_result.json")
    assert timeline_result["aggregate"]["step_count"] == 2
    assert (runs_root / run_id / "timeline_report.md").exists()

    script_path = tmp_path / "memory_agent.py"
    script_path.write_text(
        """
import json
import sys
from pathlib import Path

from sdt_bench.knowledge.chunking import build_doc_chunks_from_directory
from sdt_bench.schemas.episode import ProgrammingEpisodeSpec
from sdt_bench.schemas.retrieval import MutationRecord
from sdt_bench.utils.hashing import sha256_text
from sdt_bench.utils.time import utc_timestamp

input_dir = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
manifest = json.loads((input_dir / "manifest.json").read_text())
episode = ProgrammingEpisodeSpec.model_validate(json.loads((input_dir / "episode.json").read_text()))
docs_manifest = json.loads((input_dir / "docs" / "manifest.json").read_text())
docs = [item["path"] for item in docs_manifest["documents"][:1]]
chunks = build_doc_chunks_from_directory(
    input_dir / "docs" / "available",
    docs,
    chunk_size=1000,
    overlap=150,
    version_tag="test",
    metadata={"episode_id": episode.episode_id},
)
memory_count = json.loads((input_dir / "memory" / "manifest.json").read_text())["chunk_count"]
output_dir.mkdir(parents=True, exist_ok=True)
(output_dir / "plan.json").write_text(json.dumps({"summary": "persist first doc", "steps": ["ingest"]}))
(output_dir / "ingestion_decision.json").write_text(json.dumps({
    "strategy": "selected_visible",
    "ingest_visible_docs": True,
    "selected_visible_doc_paths": docs,
    "acquisitions": docs,
    "reason": "persist the first doc"
}))
(output_dir / "retrieval_decision.json").write_text(json.dumps({"query": "", "top_k": 0, "reason": "none"}))
(output_dir / "retrieval_trace.json").write_text(json.dumps({
    "episode_id": manifest["episode_id"],
    "query": "",
    "retrieved_chunk_ids": [],
    "retrieved_document_ids": [],
    "scores": [],
    "freshness_labels": [],
    "timestamp": utc_timestamp()
}))
(output_dir / "patch.diff").write_text("")
(output_dir / "citations.json").write_text(json.dumps({"citations": docs}))
mutations = []
if chunks and memory_count == 0:
    chunk = chunks[0]
    mutations.append({
        "record_id": sha256_text(f"{episode.episode_id}:insert:{chunk.chunk_id}"),
        "episode_id": episode.episode_id,
        "operation": "insert",
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "source_path": chunk.source_path,
        "old_hash": None,
        "new_hash": chunk.content_hash,
        "timestamp": utc_timestamp(),
        "reason": "persist first doc",
    })
with (output_dir / "memory_mutations.jsonl").open("w", encoding="utf-8") as handle:
    for row in mutations:
        handle.write(json.dumps(row))
        handle.write("\\n")
(output_dir / "review.json").write_text(json.dumps({"summary": f"memory={memory_count}", "concerns": []}))
""".strip(),
        encoding="utf-8",
    )

    persistent = runner.invoke(
        app,
        [
            "run-timeline",
            str(toy_timeline_path()),
            "--agent",
            "external",
            "--agent-command",
            f'"{sys.executable}" "{script_path}" "{{input_dir}}" "{{output_dir}}"',
            "--memory-mode",
            "persistent",
        ],
    )
    assert persistent.exit_code == 0, persistent.stdout
    external_runs = tmp_path / "runtime" / "runs" / "toy" / "external"
    persistent_run_id = (external_runs / "last_run.txt").read_text(encoding="utf-8").strip()
    step_two_input = read_json(external_runs / persistent_run_id / "steps" / "001__episode_0002" / "input" / "memory" / "manifest.json")
    assert step_two_input["chunk_count"] > 0

    reset = runner.invoke(
        app,
        [
            "run-timeline",
            str(toy_timeline_path()),
            "--agent",
            "external",
            "--agent-command",
            f'"{sys.executable}" "{script_path}" "{{input_dir}}" "{{output_dir}}"',
            "--memory-mode",
            "reset",
            "--run-id",
            "run_reset",
        ],
    )
    assert reset.exit_code == 0, reset.stdout
    reset_step_two_input = read_json(external_runs / "run_reset" / "steps" / "001__episode_0002" / "input" / "memory" / "manifest.json")
    assert reset_step_two_input["chunk_count"] == 0
