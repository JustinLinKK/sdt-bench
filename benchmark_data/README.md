# Benchmark Data

`benchmark_data/` is intentionally separate from the benchmark runtime code.

It contains:

- `manifest.yaml` for top-level dataset discovery
- `timelines/` for ordered repo timelines
- `states/` for immutable software states
- `events/` for dependency-drift transitions
- `episodes/` for agent-facing step definitions
- `fixtures/` for local synthetic repos
- `authoring/` for generated release streams, event streams, and historical snapshots

This split makes it easier for other teams to use the benchmark with their own
agent framework without depending on the bundled reference agents.
