# Architecture

`clab` is organized around a replayable episode lifecycle:

1. load and validate a self-contained episode
2. materialize a working repository snapshot
3. ingest visible external docs into a vector DB backend
4. run an agent against a constrained execution context
5. evaluate the resulting patch and traces
6. emit structured artifacts and a human-readable report

## Module boundaries

- `schemas/`: typed benchmark contracts and result models
- `benchmark/`: episode loading, validation, and materialization helpers
- `repos/`: repository-specific adapters
- `env/`: workspace, install, patch, and test execution helpers
- `knowledge/`: chunking, ingestion, freshness, citations, mutation logging
- `vectordb/`: swappable vector DB abstraction
- `agents/`: agent protocol plus bundled baselines
- `execution/`: orchestration for retrieval, patching, and review
- `evaluation/`: metrics, scoring, hidden test execution, reporting
- `utils/`: filesystem, hashing, subprocess, git, time, and logging helpers

## Episode flow

Each episode is self-contained. The only new knowledge the agent can ingest in
v0 is the content under `visible_docs/`. Hidden evaluation files remain available
to the harness but are never exposed to the agent context.

## Agent interface

Agents are adapter-based. The benchmark harness controls the workflow while the
agent contributes decisions for planning, ingestion, retrieval, patch generation,
and review. This keeps the benchmark reusable across future models.

## Vector DB update pipeline

Visible docs are chunked deterministically, assigned stable document and chunk
identifiers, and written through the backend abstraction. Every insert, update,
delete, or tombstone is captured as a mutation log entry for later scoring.

## Evaluation pipeline

Patch application, visible tests, hidden tests, retrieval freshness, mutation-log
quality, citation overlap, and code churn are computed separately before being
combined into a weighted final score.

