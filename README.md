# sdt-bench

`sdt-bench` is a research-grade scaffold for benchmarking continual-learning coding
agents under dependency version drift. It focuses on replayable episodes, explicit
vector database mutation logs, clean agent interfaces, and auditable evaluation.

## Motivation

Modern coding agents often need fresh dependency knowledge after a repository has
already been frozen. `sdt-bench` models that lifecycle directly:

1. freeze a repo at `t0`
2. introduce a dependency upgrade event at `t1`
3. reveal new external knowledge through visible docs
4. ingest those docs into an external memory backend
5. retrieve fresh knowledge, patch the codebase, and evaluate the outcome

The benchmark tracks not only whether the code works after the upgrade, but also
whether the external memory was updated correctly, whether the retrieved knowledge
was fresh, and how much code churn was introduced.

## Current status

This repository is the v0 scaffold. It includes:

- full end-to-end support for one synthetic `requests` episode
- a benchmark core package plus a separate bundled baseline-agent package
- in-memory and embedded Qdrant vector DB backends
- deterministic chunking and mutation logging
- authoring commands for release harvesting, event generation, snapshot replay, artifact synthesis, and aggregation
- CLI commands for validation, materialization, ingestion, execution, evaluation, and reporting
- documentation, tests, Docker support, and CI workflows

The `pytest` repo is partially scaffolded, while `sphinx`, `sqlfluff`, and
`xarray` are placeholder configs for future expansion.

## Repository layout

```text
.
├── benchmark_data/     # benchmark episodes and authoring outputs
├── configs/            # global and per-repo config
├── docs/               # benchmark and implementation documentation
├── src/sdt_bench/      # benchmark core package
├── src/sdt_bench_baselines/ # bundled reference agents
├── tests/              # unit and smoke integration tests
├── .github/workflows/  # CI and benchmark smoke runs
├── Dockerfile          # reproducible runtime image
└── Makefile            # install/lint/test helpers
```

## Quickstart

```bash
make install
make test
python -m sdt_bench.cli validate-episode benchmark_data/episodes/requests/episode_0001
python -m sdt_bench.cli materialize benchmark_data/episodes/requests/episode_0001
python -m sdt_bench.cli ingest-visible-docs benchmark_data/episodes/requests/episode_0001
python -m sdt_bench.cli run-agent benchmark_data/episodes/requests/episode_0001 --agent baseline:dummy
python -m sdt_bench.cli evaluate benchmark_data/episodes/requests/episode_0001
python -m sdt_bench.cli report benchmark_data/episodes/requests/episode_0001
```

For the fast local smoke test path, run:

```bash
make smoke
```

## Authoring

`sdt-bench` now includes first-pass authoring utilities for scaling beyond hand-authored episodes:

```bash
sdt-bench author-harvest-releases --repo-name requests
sdt-bench author-build-events --repo-name requests
sdt-bench author-materialize-snapshot --repo-name requests --ref refs/tags/v2.31.0
sdt-bench author-synthesize-artifacts benchmark_data/episodes/requests/episode_0001
sdt-bench aggregate-results --results-root runs
```

## Using your own agent framework

The benchmark core is separate from the bundled baselines. You can:

- use a bundled reference agent such as `baseline:dummy`, `baseline:retrieval_baseline`, `baseline:static_rag`, or `baseline:full_reindex`
- load a Python agent factory with `--agent-factory module.path:callable`
- run an external framework with `--agent-command` and exchange files through `agent_context.json` and `external_agent_output/`

## Supported repos

- `requests` - full v0 example
- `pytest` - partial scaffold
- `sphinx` - placeholder
- `sqlfluff` - placeholder
- `xarray` - placeholder

## Documentation

- [docs/index.md](docs/index.md)
- [docs/architecture.md](docs/architecture.md)
- [docs/quickstart.md](docs/quickstart.md)
- [docs/benchmark_spec.md](docs/benchmark_spec.md)
- [docs/episode_schema.md](docs/episode_schema.md)
- [docs/agent_interface.md](docs/agent_interface.md)
- [docs/authoring.md](docs/authoring.md)
- [docs/vector_db_protocol.md](docs/vector_db_protocol.md)
- [docs/evaluation.md](docs/evaluation.md)
- [docs/repo_onboarding.md](docs/repo_onboarding.md)
- [docs/development.md](docs/development.md)

## Limitations of v0

- only one full benchmark episode is provided out of the box
- the bundled baselines are intentionally simple reference agents, not competitive systems
- hidden evaluation for the bundled `requests` episode is intentionally narrow and synthetic
- Docker support is provided for reproducibility, but the default local flow runs on the host for speed
