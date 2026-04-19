# Phase-1 Pilot — Continuous-Learning Benchmark Under Dependency Drift

Implementation of the Python-only pilot described in the paper
`Controlled-Environment Benchmark for Continuous-Learning Code Agents
Under Dependency Drift` (project root).

## Layout

```
benchmark/pilot/
  src/              # 10 reference scripts
  baselines/        # static_rag, full_reindex
  config/repos.yaml # 5 Python target repos + model selection
  data/             # derived artefacts (gitignored)
    releases/       # harvest_releases.py output
    events/         # build_event_stream.py output
    snapshots/      # materialise_snapshot.py output (+ chunks.jsonl)
    gold_logs/      # apply_gold_db_diff.py output
    vector_db/      # Qdrant local store
    runs/           # per-agent-run outputs + metrics
  record.md         # experiment log (per global CLAUDE.md rule)
```

## Run order

```
source .venv/bin/activate
python src/harvest_releases.py          # pull versions + OSV advisories
python src/build_event_stream.py        # bucket into patch/minor/major/security
python src/materialise_snapshot.py      # clone & checkout tags
python src/chunk_and_ingest.py --dry-run    # build chunks.jsonl (no Qdrant embedding)
python src/chunk_and_ingest.py              # with Qdrant upsert (uses OpenRouter key)
python src/apply_gold_db_diff.py        # gold insert/update/delete/tombstone log
python src/run_agent.py                 # agent proposes mutations + patch/QA
python src/score_patch.py
python src/score_review.py              # phase-2 stub
python src/measure_diff.py
python src/aggregate_metrics.py         # final metrics.json
```

## Metrics produced

Written to `data/runs/<run_id>/metrics.json`:
- `success_at_t_overall`
- `auac_per_package` — area under adaptation curve
- `update_f1_mean_micro` — agent mutation log vs gold
- `regression_rate` — version-aware but spec broken
- `update_f1_per_event` — per (pkg, from→to)

## Configuration

Edit `config/repos.yaml` to change repos, event cap, acquisition budget,
or swap the OpenRouter agent model. Free-tier defaults:
- agent: `meta-llama/llama-3.3-70b-instruct:free`
- fallback: `google/gemini-2.0-flash-exp:free`
- embedding: `openai/text-embedding-3-small`

## Env

Expects `OPENROUTER_API_KEY` in `.env.macos` at the repo root. If unset,
`run_agent.py` falls back to a deterministic heuristic agent so the
pipeline still produces scorable output.
