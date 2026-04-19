# Agent Interface

Agents integrate through the `BaseAgent` protocol. The lifecycle is:

1. `plan(context)` - produce a short plan
2. `ingest(context)` - declare ingestion intent
3. `retrieve(context)` - declare retrieval query/top-k behavior
4. `generate_patch(context)` - emit a unified diff proposal plus citations
5. `review(context, proposal)` - summarize the patch quality or limitations

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

## Integration guidance

Future Codex or external model adapters should remain thin. The benchmark harness
owns state transitions, artifact writing, and evaluation. Adapters should focus on
turning the constrained agent context into a patch proposal and review output.

