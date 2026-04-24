# Authoring

`sdt-bench` separates the benchmark harness from benchmark-data authoring. The
authoring commands provide a migration path from the old pilot scripts into the
current typed episode model.

## Commands

- `author-harvest-releases --project-id <id>` fetches framework package release history and optional OSV advisories.
- `author-build-events --project-id <id>` converts harvested releases into an ordered dependency-event stream.
- `author-materialize-snapshot --project-id <id> --ref <git-ref>` creates a replayable upstream framework snapshot manifest.
- `author-synthesize-artifacts <episode-path>` generates `gold_mutations.yaml` and `expected_retrieval_chunks.yaml`.
- `aggregate-results --results-root runs` summarizes benchmark runs and computes longitudinal metrics such as AUAC-like per-project curves.

## Data layout

Authoring outputs live under `benchmark_data/authoring/`:

- `releases/`
- `events/`
- `snapshots/`

Benchmark projects live under `benchmark_data/projects/<project_id>/`.

## Relationship to the pilot

These commands port the highest-value ideas from the old pilot:

- release harvesting
- event stream construction
- historical snapshot replay
- gold artifact synthesis
- run aggregation

They are integrated into the current episode-oriented framework rather than being standalone scripts.
