# sdt-bench

`sdt-bench` is a timeline-first benchmark scaffold for coding agents under dependency drift.
Instead of treating each task as an isolated issue, it models software maintenance as a temporal
backtest: each step exposes only the repository, docs, failure signal, and external memory that
would have been visible at that simulated time.

## What Changed

The benchmark now centers on four immutable dataset objects:

- `TemporalStateSpec`
- `DependencyEventSpec`
- `ProgrammingEpisodeSpec`
- `TimelineSpec`

The scored unit is still a single transition episode, but the primary execution mode is now
`run-timeline`, which rematerializes code at each step, carries forward memory only, and writes a
standardized filesystem bundle for every agent interaction.

## Repository Layout

```text
benchmark_data/
  manifest.yaml
  timelines/
  states/
  events/
  episodes/
  fixtures/
configs/
docs/
src/sdt_bench/
src/sdt_bench_baselines/
tests/
```

## Quickstart

```bash
make install
make test
python -m sdt_bench.cli validate-step benchmark_data/episodes/toy/episode_0001
python -m sdt_bench.cli run-timeline benchmark_data/timelines/toy.yaml --agent baseline:dummy
python -m sdt_bench.cli report-timeline benchmark_data/timelines/toy.yaml --agent baseline:dummy
```

The committed `toy` timeline is fully offline and is the default smoke path. A scaffolded
`requests` timeline is included to show the real-repo layout expected by the benchmark.

## Agent Contract

Each step writes a bundle under:

```text
runs/<timeline_id>/<agent_slug>/<run_id>/steps/<index>__<episode_id>/
  input/
  output/
  harness/
```

The harness owns `input/` and `harness/`. Agents are expected to read only `input/` and write all
required artifacts under `output/`.

## Included Tracks

- `toy`: fully offline synthetic timeline used by tests and CI smoke runs
- `requests`: scaffolded real-repository timeline showing the intended production layout

## Documentation

- [docs/index.md](docs/index.md)
- [docs/architecture.md](docs/architecture.md)
- [docs/quickstart.md](docs/quickstart.md)
- [docs/benchmark_spec.md](docs/benchmark_spec.md)
- [docs/episode_schema.md](docs/episode_schema.md)
- [docs/agent_interface.md](docs/agent_interface.md)
- [docs/vector_db_protocol.md](docs/vector_db_protocol.md)
- [docs/evaluation.md](docs/evaluation.md)
