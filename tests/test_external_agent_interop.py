from __future__ import annotations

import sys
from pathlib import Path

from sdt_bench.integrations.external_agents import run_external_agent_command
from sdt_bench.schemas import AgentContext, EpisodeBudget, EpisodeSpec, RepoSpec
from sdt_bench.schemas.event import DependencyEvent


def test_run_external_agent_command(tmp_path: Path) -> None:
    script_path = tmp_path / "agent.py"
    script_path.write_text(
        """
import json
import sys
from pathlib import Path

context_path = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
context = json.loads(context_path.read_text())
output_dir.mkdir(parents=True, exist_ok=True)
(output_dir / "patch.diff").write_text("")
(output_dir / "citations.json").write_text(json.dumps({"citations": context["available_visible_doc_paths"][:1]}))
(output_dir / "review.json").write_text(json.dumps({"summary": "external ok", "concerns": []}))
(output_dir / "mutation_log.jsonl").write_text("")
""".strip(),
        encoding="utf-8",
    )
    context = AgentContext(
        episode=EpisodeSpec(
            episode_id="episode_0001",
            repo_name="toy",
            base_commit="abcdef0",
            base_ref="refs/heads/main",
            dependency_event=DependencyEvent(
                event_id="event",
                dependency_name="toydep",
                ecosystem="pypi",
                old_version="0.1.0",
                new_version="0.2.0",
                event_type="synthetic",
                summary="synthetic",
                breaking_change_expected=False,
                visible_doc_paths=["visible_docs/changelog.md"],
                gold_mutation_paths=[],
                metadata={},
            ),
            task_family="smoke",
            task_prompt="test",
            visible_failure_signal="signal",
            visible_doc_paths=["visible_docs/changelog.md"],
            hidden_test_command="echo",
            hidden_test_manifest="hidden_eval/tests_manifest.yaml",
        ),
        repo_spec=RepoSpec(
            name="toy",
            github_url="https://example.com/toy.git",
            package_name="toy",
            ecosystem="PyPI",
            default_branch="main",
            language="python",
            package_manager="pip",
            install_command="uv pip install -e .",
            test_command="python -m pytest",
        ),
        workspace=str(tmp_path),
        run_dir=str(tmp_path),
        task_prompt="test",
        visible_failure_signal="signal",
        available_visible_doc_paths=["visible_docs/changelog.md"],
        visible_chunks_path=None,
        retrieved_chunks=[],
        budget=EpisodeBudget(max_runtime_seconds=10, retrieval_top_k=1, max_visible_docs=1, acquisition_budget=1),
        backend_name="in_memory",
    )
    command = f'"{sys.executable}" "{script_path}" "{{context_json}}" "{{output_dir}}"'
    plan, ingestion, retrieval, trace, proposal, review, mutations = run_external_agent_command(
        context=context,
        command_template=command,
        run_root=tmp_path,
    )
    assert plan.summary
    assert ingestion.strategy == "none"
    assert retrieval.query == ""
    assert proposal.citations_used == ["visible_docs/changelog.md"]
    assert review.summary == "external ok"
    assert mutations == []
