# sdt-bench

`sdt-bench` is a timeline-first benchmark scaffold for coding agents under dependency drift.
Instead of treating each task as an isolated issue, it models software maintenance as a temporal
backtest: each step exposes only the repository, docs, failure signal, and external memory that
would have been visible at that simulated time.

## Mental Model

The benchmark is built from four dataset objects:

- `TemporalStateSpec`: what the repo world looks like at one timestamp
- `DependencyEventSpec`: what changed between two states
- `ProgrammingEpisodeSpec`: the agent task created by that change
- `TimelineSpec`: the ordered history for one repository

The easiest way to think about them is:

- a `state` is a point on the timeline
- an `event` is the drift between two points
- an `episode` is the repair task on that edge
- a `timeline` is the full ordered sequence

```text
time --------------------------------------------------------------->

State S0                  State S1                  State S2
  |                         |                         |
  |---- Episode 0001 ------>|---- Episode 0002 ----->|
         caused by Event 1           caused by Event 2
```

An episode does not mean "the whole repo benchmark". It means one transition:

- start from `from_state`
- adapt to `to_state`
- use only docs visible by `to_state.timestamp`
- solve the failure for that step

That is why the benchmark feels like a maintenance backtest rather than a static bug-fix suite.

## Scope

The framework supports many repositories, but each individual timeline belongs to one repo.

```text
benchmark
  repo A
    timeline A1
      episode 1, 2, 3...
  repo B
    timeline B1
      episode 1, 2, 3...
```

So:

- one `episode` = one repo
- one `timeline` = one repo
- the overall framework = can host many repos

## Execution Model

The scored unit is a single `episode`, but the main execution mode is `run-timeline`.

For each step the harness:

1. loads the episode and its linked `from_state`, `to_state`, `event`, and `timeline`
2. creates a fresh step directory
3. clones the repo and checks out the `from_state` commit
4. installs the environment declared by `to_state`
5. stages only the docs visible at `to_state.timestamp`
6. mounts the current memory snapshot
7. lets the agent read `input/` and write outputs under `output/`
8. applies the patch, runs visible and hidden tests, and scores the step

Important persistence rule:

- code does not carry from one episode to the next
- memory can carry forward when `--memory-mode persistent` is used

This means each episode is isolated at the filesystem and repo-state level, while memory is the
only intentional cross-step channel.

## Repository Layout

```text
benchmark_data/
  manifest.yaml
  timelines/<repo>.yaml
  states/<repo>/<state_id>/
  events/<repo>/<event_id>/
  episodes/<repo>/<episode_id>/
  fixtures/
configs/
docs/
src/sdt_bench/
src/sdt_bench_baselines/
tests/
```

Each repo contributes a timeline and three families of objects:

```text
benchmark_data/
  timelines/
    toy.yaml
  states/
    toy/
      toy_2026_01/
      toy_2026_02/
      toy_2026_03/
  events/
    toy/
      toy_event_0001/
      toy_event_0002/
  episodes/
    toy/
      episode_0001/
      episode_0002/
```

In the `toy` example, the alignment is:

```text
toy_2026_01 --(toy_event_0001 / episode_0001)--> toy_2026_02
toy_2026_02 --(toy_event_0002 / episode_0002)--> toy_2026_03
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

Each step writes a bundle under a run directory:

```text
runs/<timeline_id>/<agent_slug>/<run_id>/steps/<index>__<episode_id>/
  input/
  output/
  harness/
```

The harness owns `input/` and `harness/`. Agents are expected to read only `input/` and write all
required artifacts under `output/`.

A typical step looks like this:

```text
steps/000__episode_0001/
  input/
    workspace/          # fresh checkout at from_state
    docs/available/     # only visible docs at this simulated time
    memory/             # carried snapshot or empty snapshot
    visible_failure/
    manifest.json
  output/
    plan.json
    ingestion_decision.json
    retrieval_decision.json
    retrieval_trace.json
    patch.diff
    citations.json
    review.json
    memory_mutations.jsonl
  harness/
    materialization.json
    environment.json
    candidate_chunks.jsonl
```

The key ownership rule is simple:

- harness-owned: `input/`, `harness/`
- agent-owned: `output/`

## What The Agent Can See

At each episode, the agent can access only:

- the repo materialized at `from_state`
- the visible failure signal for that episode
- docs whose `available_at` is not later than `to_state.timestamp`
- the current serialized memory snapshot
- step metadata under `input/*.json`

The agent cannot see:

- hidden evaluation files
- future docs
- future states or future events
- gold mutation or expected retrieval artifacts

This is the core temporal guardrail that keeps the benchmark historically consistent.

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
