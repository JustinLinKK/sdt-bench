# Benchmark Spec

## First-class objects

- `TemporalStateSpec`: immutable repository state, dependency snapshot, environment, and visible state docs
- `DependencyEventSpec`: drift event linking `from_state` to `to_state`
- `ProgrammingEpisodeSpec`: agent-facing transition task
- `TimelineSpec`: ordered set of states and episodes for one repo

## What the agent sees

At each step the agent can access only:

- the materialized workspace for `from_state`
- the filtered docs staged under `input/docs/available/`
- the visible failure signal
- the current memory snapshot under `input/memory/`
- the step metadata written to `input/*.json`

## What stays hidden

- `hidden_eval/`
- gold mutation artifacts
- expected retrieval artifacts
- any future state or event docs

## Primary evaluation modes

- `persistent`: carry memory forward across steps
- `reset`: clear memory before every step

The benchmark reports both per-step metrics and timeline-level aggregates such as adaptation area,
mean time to recover, stale retrieval rate, and drawdown.
