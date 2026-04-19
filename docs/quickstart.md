# Quickstart

## Install

```bash
make install
```

## Validate One Step

```bash
python -m sdt_bench.cli validate-step benchmark_data/episodes/toy/episode_0001
```

## Run A Full Timeline

```bash
python -m sdt_bench.cli run-timeline benchmark_data/timelines/toy.yaml --agent baseline:dummy
python -m sdt_bench.cli report-timeline benchmark_data/timelines/toy.yaml --agent baseline:dummy
```

## Step-by-step Flow

```bash
python -m sdt_bench.cli materialize-step benchmark_data/episodes/toy/episode_0001 --agent baseline:dummy
python -m sdt_bench.cli run-step benchmark_data/episodes/toy/episode_0001 --agent baseline:dummy
python -m sdt_bench.cli evaluate-step benchmark_data/episodes/toy/episode_0001 --agent baseline:dummy
```

## Output Layout

```text
runs/<timeline_id>/<agent_slug>/<run_id>/
  steps/<index>__<episode_id>/
    input/
    output/
    harness/
  timeline_result.json
  timeline_report.md
```
