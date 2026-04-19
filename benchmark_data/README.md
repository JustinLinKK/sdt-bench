# Benchmark Data

`benchmark_data/` is intentionally separate from the benchmark runtime code.

It contains:

- `episodes/` for self-contained benchmark episodes
- `authoring/` for generated release streams, event streams, and historical snapshots

This split makes it easier for other teams to use the benchmark with their own
agent framework without depending on the bundled reference agents.
