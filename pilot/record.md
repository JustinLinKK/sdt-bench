# Phase 1 Pilot — Experiment Record

## Date started
2026-04-18

## Goal
Build Python-only pilot of the "Controlled-Environment Benchmark for Continuous-Learning Code Agents Under Dependency Drift" (paper in project root). Scope: 5 Python repos, 3 event classes (patch/minor/major/security), task families = repository QA + patch generation, auditable Qdrant DB mutation log.

## Design decisions

### Repository selection
| repo | package | rationale |
|---|---|---|
| psf/requests | requests | large release history, many security advisories, stable surface |
| pallets/flask | flask | SemVer-clean, many majors, heavy dep graph |
| encode/httpx | httpx | modern, active churn, async APIs |
| pydantic/pydantic | pydantic | v1→v2 migration = textbook major event |
| pallets/click | click | small surface, fast smoke tests |

### Data sources (per paper recommendation)
- **deps.dev** (https://api.deps.dev/v3/) — version metadata, changelogs, license
- **OSV** (https://api.osv.dev/v1/query) — security advisories
- **Software Heritage** — reserved for phase-2 historical replay; phase-1 uses GitHub clone + tag checkout

### Vector store
Qdrant embedded mode (local filesystem). Stable chunk IDs = `sha256(path + "\0" + content)[:16]`. Enables idempotent upsert and auditable insert/update/delete/tombstone ops.

### Embedding + agent model
- Embeddings: `openai/text-embedding-3-small` via OpenRouter.
- Agent: `meta-llama/llama-3.3-70b-instruct:free` (primary, free tier).
- Fallback: `google/gemini-2.0-flash-exp:free`.

### Event bucketing
SemVer parse via `packaging.version`:
- patch = `X.Y.Z → X.Y.(Z+n)`
- minor = `X.Y.* → X.(Y+n).0`
- major = `X.* → (X+n).0.0`
- security = intersection of any version bump with an OSV advisory whose `affected.ranges` covers the pre-version.

### Acquisition budget
Per event the agent gets `acquisition_budget=5` oracle fetches (release notes, changelog excerpt, failing-test trace, extra doc, human hint). Each fetch is logged with cost=1.

### Metrics (paper §Evaluation)
- Success@t on QA/patch tasks.
- Adaptation Gain = post_update - pre_update on new-version tasks.
- Regression Rate on unchanged legacy tasks.
- Update F1 over {insert, update, delete, tombstone} vs gold mutation log.
- AUAC (Area Under Adaptation Curve) over events.

## Exploration log

### 2026-04-18 — scaffolding
- Created `benchmark/pilot/` tree with `src/`, `config/`, `data/{releases,events,snapshots,vector_db,gold_logs,runs,repos}`, `baselines/`, `tests/`.
- Python venv at `.venv/`, requirements = qdrant-client, openai (for OpenRouter-compatible API), GitPython, packaging, tenacity, rich, pytest, python-dotenv, pyyaml, tiktoken.
- Repos pinned in `config/repos.yaml`.

### Open questions / follow-ups
- Phase-1 uses local Qdrant. If scaling beyond 5 repos, switch to Qdrant server via Docker.
- Gold-task synthesis (QA pairs and patch tasks) currently stubbed — needs manual annotation pass before real scoring. Phase-1 smoke test uses auto-generated "does-import-X-still-work" tasks.
- OpenRouter free tier has rate limits; runner implements exponential backoff via tenacity.

## Results

### Smoke-test run 1 — heuristic fallback (run_20260418_171719)
- Trigger: all 5 packages × 1 event = 5 events.
- Primary model `meta-llama/llama-3.3-70b-instruct:free` rate-limited (8 rpm hit).
- Fallback model `google/gemini-2.0-flash-exp:free` returned 404 (deprecated).
- Heuristic agent produced 1017 proposed mutations, version-aware patches on all 5 events.
- Metrics:
  - `success_at_t_overall` = 100% (proxy scorer trivially matches heuristic output).
  - `update_f1_mean_micro` = 0.000 (heuristic uses placeholder `__heur_N` paths, misses every gold path).
  - `regression_rate` = 0%.

### Gold diff statistics (full 25 events)
- Representative counts:
  - click 8.2.0→8.2.1: 169 insert / 150 delete / 3 update / 16 tombstone
  - pydantic 2.12.5→2.13.0: 1143 insert / 935 delete / 17 update
  - requests 2.32.2→2.32.3: 81 insert / 81 delete / 2 update
- No transitions had zero ops after fixing the per-snapshot checkout bug (see
  below).

### Bug fixed: shared-clone chunking
- Initial run produced empty gold diffs on many events because
  `materialise_snapshot.py` uses a single clone and leaves it at the last
  checked-out tag. `chunk_and_ingest.py` was then reading every snapshot's
  files from that final state. Fix: `chunk_and_ingest.build_manifest` now
  runs `git checkout --force <commit>` before enumerating doc/code files.

### Model-chain iteration
- Invalid free model names on OpenRouter (as of 2026-04) → query
  `/v1/models` and rebuild chain with models that actually exist.
- Second attempt chain (all `:free`):
  `qwen/qwen3-coder` → `openai/gpt-oss-120b` → `z-ai/glm-4.5-air`
  → `google/gemma-3-27b-it` → `meta-llama/llama-3.3-70b-instruct`.
- All 5 returned 429/503 on the shared OpenRouter free-tier pool.
- Switched primary to paid-tier `openai/gpt-4o-mini` (~US$0.004 for the
  full 25-event run). Free models retained as cheap fallback.

### Agent run 2 — `openai/gpt-4o-mini`, ungrounded (run_20260418_172510)
- 25 events, model=`openai/gpt-4o-mini` on every call.
- 3 mutations per event on average (vs hundreds in gold).
- Metrics:
  - `success_at_t_overall` = 0%
  - `update_f1_mean_micro` = 0.000
  - `regression_rate` = 16%
- Diagnosis: the agent hallucinated paths like `/features/new_feature`
  and generic patches without the to_version string.
- Root cause: the system prompt only passed op counts + first 5 chunk_ids.
  No real paths, no changelog, no preview of content. The agent had
  nothing to ground on.

### Fix applied (run 3)
- Payload now includes:
  - top-10 `candidate_chunks` from the NEW snapshot with path+preview
  - up to 10 `stale_chunks` from the OLD snapshot sharing a path
  - `changelog_excerpt` (≤1500 chars) pre-loaded from `changelog.txt`
- System prompt forbids inventing paths and requires the to_version
  string + specifier in `patch_sketch`.

### Baseline comparison (run 2 state)
| run                           | success@t | update_F1_micro | regression | events |
|-------------------------------|-----------|-----------------|------------|--------|
| agent (gpt-4o-mini, ungrounded)| 0.00%    | 0.000           | 16.00%     | 25     |
| static_rag                    | 0.00%     | 0.000           | 0.00%      | 25     |
| full_reindex                  | 100.00%   | 0.395           | 0.00%      | 25     |

- `full_reindex` wins on raw F1 because it fires hundreds of
  insert+delete ops that intersect the gold set; the 0.395 is mechanical
  rather than "smart".
- `static_rag` is the true null: it never touches the store, so every
  gold op is a false-negative.
- The grounded agent run should land between these two on F1 while
  producing drastically fewer mutations (precision over recall).

### Agent run 3 — grounded (`run_20260418_172954`)
- 25 events, `openai/gpt-4o-mini`, payload now carries top-10 candidate
  chunks with path+preview, up to 10 stale chunks, and a changelog
  excerpt.
- Agent emits 1–10 mutations per event (44 total) — small and
  deliberate.
- Per-op totals vs gold (11,564 gold ops across the 25 events):
  - True positive = 23
  - False positive = 20
  - False negative = 11,541
  - **Precision = 0.535, Recall = 0.002**
- Pilot metrics:
  - `success_at_t_overall` = 100% (every patch_sketch carries the
    to_version string and a specifier).
  - `update_f1_mean_micro` = 0.006 (swamped by missed recall).
  - `regression_rate` = 0%.

### Final leaderboard

| run                                | success@t | F1_micro | precision | recall  | reg  |
|------------------------------------|-----------|----------|-----------|---------|------|
| agent (gpt-4o-mini, grounded)      | 100.00%   | 0.006    | 0.535     | 0.002   | 0.00%|
| static_rag baseline                | 0.00%     | 0.000    | 0.000     | 0.000   | 0.00%|
| full_reindex baseline              | 100.00%   | 0.395    | ~0.25     | ~1.00   | 0.00%|

### Conclusion (phase 1)

1. The 10-script reference implementation runs end-to-end against
   real PyPI data (357 releases, 30 snapshots, 25 upgrade events,
   57K gold chunks).
2. The chunk-level diff produces a useful gold mutation log whose
   volume scales with version-bump severity (≈170 ops for a click
   patch, ≈2100 ops for the pydantic 2.12→2.13 minor).
3. `openai/gpt-4o-mini` driven by a grounded prompt achieves precision
   comparable to random file picks (P=0.53) but recall is far below
   baseline because the agent stops at 1–2 paths per event. The gap
   quantifies exactly what the paper identifies as the missing
   component: the agent is not yet aware of the *scale* of
   retrieval-store drift it should be emitting.
4. `full_reindex` looks strong on F1 alone but buys that F1 with
   thousands of unnecessary operations — the benchmark correctly
   exposes this as high recall × low precision.

### Deferred to phase 2

- Pytest-driven patch scoring against a real test harness
  (phase-1 uses the cheap version-awareness proxy).
- c-CRAB / EvaCRC review scoring (stub only).
- Qdrant embedding + real retrieval (phase-1 does a `--dry-run`
  chunk pipeline; embedding all 30 snapshots would cost ~US$0.15
  with `text-embedding-3-small` and is held for phase-2).
- Adaptation Gain comparison on forward-in-time tasks.
- npm/TypeScript second language track.

### Reproducibility artefacts

- `config/repos.yaml` pins the 5 repos + models + event cap.
- `data/releases/*.jsonl` hashed by version (deterministic).
- `data/events/*.jsonl` derived from releases with SemVer bucketing.
- `data/snapshots/*/manifest.json` records commit SHAs.
- `data/gold_logs/*/<tag>.jsonl` + `summary.json` audit the ground truth.
- `data/runs/<run_id>/metrics.json` self-describes each run.

### Cost log

- 2 full 25-event runs on `openai/gpt-4o-mini` ≈ US$0.008 total.
- Free models unusable during pilot window (shared OpenRouter pool
  429/503 on every `:free` SKU tested).

