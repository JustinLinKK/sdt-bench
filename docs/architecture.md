# Architecture

`sdt-bench` now treats dependency-drift maintenance as a temporal simulation.

## Core flow

1. Load a `ProgrammingEpisodeSpec` and resolve its `from_state`, `to_state`, `event`, and `timeline`.
2. Materialize the project snapshot from `from_state` and install the environment declared by `to_state`.
3. Stage only the documents visible at `to_state.timestamp` into `input/docs/available/`.
4. Mount the current memory snapshot under `input/memory/`.
5. Let the agent read `input/` and write required artifacts under `output/`.
6. Apply the proposed patch, run visible and hidden tests, score the step, and update memory for the next step.

## Module boundaries

- `schemas/`: typed contracts for states, events, episodes, timelines, I/O manifests, and results
- `benchmark/`: object loading, validation, visibility filtering, and step materialization
- `execution/`: agent execution for one step
- `evaluation/`: step scoring, temporal aggregation, and reports
- `knowledge/`: chunking, mutation tracking, and persistent-memory application
- `env/`: workspace setup, offline installation, patching, and test execution

## Persistence model

- Code does not persist across steps.
- Memory does persist when `--memory-mode persistent` is used.
- Hidden evaluation artifacts stay under the harness-owned `projects/<project_id>/episodes/<episode_id>/hidden_eval/` tree and are never copied into `input/`.
