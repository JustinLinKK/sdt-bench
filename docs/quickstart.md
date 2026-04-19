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
python -m sdt_bench.cli validate-episode benchmark_data/episodes/requests/episode_0001
python -m sdt_bench.cli materialize benchmark_data/episodes/requests/episode_0001
python -m sdt_bench.cli ingest-visible-docs benchmark_data/episodes/requests/episode_0001
python -m sdt_bench.cli run-agent benchmark_data/episodes/requests/episode_0001 --agent baseline:dummy
python -m sdt_bench.cli evaluate benchmark_data/episodes/requests/episode_0001
python -m sdt_bench.cli report benchmark_data/episodes/requests/episode_0001
```

## Docker

```bash
docker build -t sdt-bench .
docker run --rm -v "$PWD:/app" sdt-bench validate-episode benchmark_data/episodes/requests/episode_0001
```

## Interpreting outputs

Each command writes artifacts under `runs/<episode_slug>/<run_id>/`. Common files:

- `materialization.json`
- `environment.json`
- `chunks.jsonl`
- `agent_context.json`
- `mutation_log.jsonl`
- `retrieval_trace.json`
- `patch.diff`
- `citations.json`
- `review.json`
- `result.json`
- `report.md`

## External agents

You can replace the bundled baselines with your own framework:

```bash
python -m sdt_bench.cli run-agent benchmark_data/episodes/requests/episode_0001 \
  --agent external \
  --agent-command "python my_agent.py {context_json} {output_dir}"
```

The benchmark writes `agent_context.json` and expects outputs in `external_agent_output/`.
