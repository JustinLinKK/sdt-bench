# Quickstart

## Local install

```bash
make install
```

This uses `uv` and installs the project plus development dependencies.

## Local smoke path

```bash
make smoke
```

The smoke test uses a tiny fixture repo and episode so it can run quickly without
network access.

## Running one benchmark episode

```bash
python -m clab.cli validate-episode data/episodes/requests/episode_0001
python -m clab.cli materialize data/episodes/requests/episode_0001
python -m clab.cli ingest-visible-docs data/episodes/requests/episode_0001
python -m clab.cli run-agent data/episodes/requests/episode_0001 --agent dummy
python -m clab.cli evaluate data/episodes/requests/episode_0001
python -m clab.cli report data/episodes/requests/episode_0001
```

## Docker

```bash
docker build -t clab .
docker run --rm -v "$PWD:/app" clab validate-episode data/episodes/requests/episode_0001
```

## Interpreting outputs

Each command writes artifacts under `runs/<episode_slug>/<run_id>/`. Common files:

- `materialization.json`
- `environment.json`
- `chunks.jsonl`
- `mutation_log.jsonl`
- `retrieval_trace.json`
- `patch.diff`
- `citations.json`
- `review.json`
- `result.json`
- `report.md`

