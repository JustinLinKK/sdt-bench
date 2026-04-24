# Agent Interface

Agents integrate through a filesystem bundle per step.

## Input bundle

```text
input/
  manifest.json
  episode.json
  from_state.json
  to_state.json
  event.json
  project_spec.json
  workspace/
  docs/manifest.json
  docs/available/
  visible_failure/ci_failure.txt
  memory/manifest.json
  memory/chunks.jsonl
```

## Required output bundle

```text
output/
  plan.json
  ingestion_decision.json
  retrieval_decision.json
  retrieval_trace.json
  patch.diff
  citations.json
  memory_mutations.jsonl
  review.json
```

Every file under `output/` is required. No-op behavior must still be encoded with valid empty
payloads.

## External command integration

Use:

```bash
python -m sdt_bench.cli run-step <episode_dir> \
  --agent external \
  --agent-command "python my_agent.py {input_dir} {output_dir}"
```

The harness also expands `{manifest_json}` and `{workspace}` for convenience.
