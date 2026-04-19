# Agent Interface

Agents integrate through the `BaseAgent` protocol. The lifecycle is:

1. `plan(context)` - produce a short plan
2. `ingest(context)` - declare ingestion intent
3. `retrieve(context)` - declare retrieval query/top-k behavior
4. `generate_patch(context)` - emit a unified diff proposal plus citations
5. `review(context, proposal)` - summarize the patch quality or limitations

The benchmark core no longer bundles concrete baseline agents directly. Reference
baselines now live in the separate `sdt_bench_baselines` package, while external
teams can provide their own agent implementations.

## Required inputs

- episode metadata
- visible failure signal
- retrieved chunks
- workspace path
- budget settings

## Required outputs

- plan metadata
- ingestion and retrieval decisions
- patch proposal
- citation list
- review result

## Integration options

- Built-in baselines: `baseline:dummy`, `baseline:retrieval_baseline`, `baseline:static_rag`, `baseline:full_reindex`
- Python factory: `--agent-factory module.path:callable`
- External process: `--agent-command "python my_agent.py {context_json} {output_dir}"`

When using `--agent-command`, the benchmark writes `agent_context.json` and expects
artifact files such as `patch.diff`, `citations.json`, `review.json`, and optional
`mutation_log.jsonl`, `plan.json`, `retrieval_decision.json`, and `retrieval_trace.json`.

## Integration guidance

Future Codex or external model adapters should remain thin. The benchmark harness
owns state transitions, artifact writing, and evaluation. Adapters should focus on
turning the constrained agent context into a patch proposal and review output.
