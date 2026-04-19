# Authoring

`sdt-bench` separates the benchmark harness from benchmark-data authoring. The
authoring commands provide a migration path from the old pilot scripts into the
current typed episode model.

## Commands

- `author-harvest-releases --repo-name <name>` fetches package release history and optional OSV advisories.
- `author-build-events --repo-name <name>` converts harvested releases into an ordered dependency-event stream.
- `author-materialize-snapshot --repo-name <name> --ref <git-ref>` creates a replayable repository snapshot manifest.
- `author-synthesize-artifacts <episode-path>` generates `gold_mutations.yaml` and `expected_retrieval_chunks.yaml`.
- `aggregate-results --results-root runs` summarizes benchmark runs and computes longitudinal metrics such as AUAC-like per-repo curves.

## Data layout

Authoring outputs live under `benchmark_data/authoring/`:

- `releases/`
- `events/`
- `snapshots/`

Episodes remain under `benchmark_data/episodes/`.

## Relationship to the pilot

These commands port the highest-value ideas from the old pilot:

- release harvesting
- event stream construction
- historical snapshot replay
- gold artifact synthesis
- run aggregation

They are integrated into the current episode-oriented framework rather than being standalone scripts.
