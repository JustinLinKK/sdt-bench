# clab

`clab` is a research-grade scaffold for benchmarking continual-learning coding
agents under dependency version drift. It focuses on replayable episodes, explicit
vector database mutation logs, clean agent interfaces, and auditable evaluation.

## Motivation

Modern coding agents often need fresh dependency knowledge after a repository has
already been frozen. `clab` models that lifecycle directly:

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
- a dummy agent and a retrieval baseline
- in-memory and embedded Qdrant vector DB backends
- deterministic chunking and mutation logging
- CLI commands for validation, materialization, ingestion, execution, evaluation, and reporting
- documentation, tests, Docker support, and CI workflows

The `pytest` repo is partially scaffolded, while `sphinx`, `sqlfluff`, and
`xarray` are placeholder configs for future expansion.

## Repository layout

```text
.
├── configs/            # global and per-repo config
├── data/episodes/      # self-contained benchmark episodes
├── docs/               # benchmark and implementation documentation
├── src/clab/           # framework package
├── tests/              # unit and smoke integration tests
├── .github/workflows/  # CI and benchmark smoke runs
├── Dockerfile          # reproducible runtime image
└── Makefile            # install/lint/test helpers
```

## Quickstart

```bash
make install
make test
python -m clab.cli validate-episode data/episodes/requests/episode_0001
python -m clab.cli materialize data/episodes/requests/episode_0001
python -m clab.cli ingest-visible-docs data/episodes/requests/episode_0001
python -m clab.cli run-agent data/episodes/requests/episode_0001 --agent dummy
python -m clab.cli evaluate data/episodes/requests/episode_0001
python -m clab.cli report data/episodes/requests/episode_0001
```

For the fast local smoke test path, run:

```bash
make smoke
```

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
- [docs/vector_db_protocol.md](docs/vector_db_protocol.md)
- [docs/evaluation.md](docs/evaluation.md)
- [docs/repo_onboarding.md](docs/repo_onboarding.md)
- [docs/development.md](docs/development.md)

## Limitations of v0

- only one full benchmark episode is provided out of the box
- the default retrieval baseline emits a no-op patch unless an external adapter is configured
- hidden evaluation for the bundled `requests` episode is intentionally narrow and synthetic
- Docker support is provided for reproducibility, but the default local flow runs on the host for speed

