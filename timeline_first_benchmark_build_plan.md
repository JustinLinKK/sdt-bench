# Timeline-First Benchmark Build Plan for a Continuous-Learning Coding Agent

**Version:** v0.1  
**Date:** 2026-04-23  
**Target audience:** coding agent / Codex-style implementation agent  
**Scope:** build three benchmark products based on Scrapy, Prefect, and Haystack

---

## 1. Goal

Build a **timeline-first benchmark** for a continuous-learning coding agent.

The benchmark should simulate a software product that evolves through a sequence of states. At each state transition:

1. the product dependency stack changes,
2. new framework or integration documentation becomes available,
3. the agent must ingest the new docs,
4. the agent must update its vector database,
5. the agent must patch the product codebase,
6. the product must pass visible and hidden tests,
7. the agent memory should persist into the next state.

This is **not** a generic issue-fixing benchmark. It is a **product evolution benchmark** where the product outcome should remain stable or improve while the frameworks, APIs, and deployment patterns evolve.

The design is intentionally analogous to **quant-finance backtesting**:

- each state is a frozen time step,
- the agent can only access documents available at that simulated time,
- hidden evaluation prevents lookahead leakage,
- performance is measured both per state and longitudinally across the whole timeline.

---

## 2. Why these three benchmark products

### 2.1 Scrapy product: `crawler_app`

Use Scrapy as the dependency framework for a crawler/extraction product.

Why this is a strong starting point:

- Scrapy is public and BSD-3-Clause licensed.
- Scrapy has explicit release notes and deprecation policy.
- Scrapy release notes contain usable migration information for API drift.
- The resulting product is easy to test with deterministic HTML fixtures and feed outputs.
- State transitions are cheaper to reproduce than heavier ML stacks.

Best benchmark themes:

- sync → async spider startup,
- reactor configuration changes,
- middleware behavior changes,
- deprecations and import path changes,
- request metadata / offsite behavior changes.

### 2.2 Prefect product: `workflow_app`

Use Prefect as the dependency framework for a workflow/data-pipeline product.

Why this is a strong second benchmark:

- Prefect is public and Apache-2.0 licensed.
- Prefect has an explicit 2.x → 3.x upgrade guide.
- Prefect 3.x introduces migration-relevant concepts: Pydantic v2, module path changes, async task behavior changes, flow final-state changes, events/automations, workers/deployments.
- This product is highly aligned with CI/CD and orchestration evolution.

Best benchmark themes:

- flow API migration,
- parameter model compatibility,
- async vs sync execution,
- deployments/workers,
- integration package upgrades.

### 2.3 Haystack product: `rag_app`

Use Haystack as the dependency framework for a retrieval / QA / RAG product.

Why this is the strongest memory-centric benchmark:

- Haystack main repo is Apache-2.0 licensed.
- Haystack has a major 1.x → 2.x migration guide.
- The migration contains deep architectural drift: `farm-haystack` → `haystack-ai`, nodes → components, pipeline rewrites, document store changes, prompt builder/generator changes, and Hayhooks replacing old REST deployment patterns.
- This product naturally overlaps with vector DB usage, retrieval freshness, and memory evaluation.

Best benchmark themes:

- pipeline architecture migration,
- generator/prompt-builder migration,
- retriever/document-store API drift,
- deployment drift,
- retrieval schema changes.

### 2.4 License / compliance note

- Scrapy: BSD-3-Clause.
- Prefect: Apache-2.0.
- Haystack: main repo is Apache-2.0; GitHub also reports an additional `license-header.txt` artifact. Treat the main framework as Apache-2.0 for dependency usage, but review redistribution details if the benchmark ships frozen code snapshots.

---

## 3. Global benchmark architecture

Build one benchmark monorepo with three benchmark products.

```text
benchmark/
  products/
    crawler_app/      # Scrapy-based product
    workflow_app/     # Prefect-based product
    rag_app/          # Haystack-based product

  states/
    scrapy/
    prefect/
    haystack/

  episodes/
    scrapy/
    prefect/
    haystack/

  registry/
    docs/
    packages/

  envs/
    scrapy/
    prefect/
    haystack/

  runs/
  reports/
```

Each product should have **4–6 states**. Recommended first version: **5 states per product**.

That yields:

- 5 frozen states per product,
- 4 transitions per product,
- 2 coding episodes per transition,
- 8 episodes per product,
- 24 episodes total across the 3 benchmark products.

---

## 4. Global state model

Each benchmark state should be immutable.

```text
State_t =
  product code snapshot
  dependency lockfile
  Docker image
  Conda/uv environment lock
  visible documentation snapshot
  vector DB snapshot
  visible tests
  hidden tests
  state metadata
```

Each state directory should contain:

```text
states/<project>/<state_id>/
  state.yaml
  product_snapshot/
  env/
    Dockerfile
    environment.yml
    conda-lock.yml
    requirements.lock
    wheelhouse/
  docs/
    corpus_manifest.yaml
    visible_docs/
  tests/
    visible/
    hidden/
  artifacts/
    golden_outputs/
```

---

## 5. Global transition / breakpoint model

A **breakpoint** is a deliberate transition where the current product code should no longer be assumed valid under the next state.

Every breakpoint should include:

```text
Breakpoint =
  dependency drift
  API or behavior change
  visible documentation delta
  expected product impact
  required vector DB update
  coding tasks triggered by the change
  hidden evaluation for the next state
```

### 5.1 Breakpoint detection logic

The coding agent runner should detect breakpoints using one or more of these signals:

1. **Dependency version change** in lockfile or manifest.
2. **Release-note / migration-guide availability** in docs registry.
3. **Import failure** or runtime error under the next environment.
4. **Behavior regression** in visible tests.
5. **Deprecation warning becomes error**.
6. **Configuration schema mismatch**.
7. **Async / sync execution mismatch**.
8. **Serialization / endpoint schema mismatch**.

The benchmark framework should explicitly record which breakpoint category applies:

```yaml
breakpoint_type:
  - api_signature_change
  - deprecation_removal
  - async_sync_semantics_change
  - config_schema_change
  - integration_api_change
  - deployment_model_change
  - dependency_package_rename
  - document_store_or_retriever_change
```

---

## 6. Shared coding-agent responsibilities at each breakpoint

At every state transition, require the coding agent to perform the following steps.

### Step A. Inspect the transition package diff

Read:

- old dependency snapshot,
- new dependency snapshot,
- visible release notes,
- visible migration docs,
- visible CI failure signal.

Output:

- dependency drift summary,
- likely impacted modules,
- likely stale vector DB areas.

### Step B. Update the vector DB

The agent must:

- ingest newly visible docs,
- chunk them deterministically,
- upsert changed chunks,
- mark outdated chunks as stale or tombstoned,
- log each mutation.

Required outputs:

- `mutation_log.jsonl`
- `retrieval_snapshot.json`
- `cited_chunk_ids.json`

### Step C. Plan required code changes

The agent must produce a structured plan such as:

```yaml
plan:
  impacted_files:
    - file_a.py
    - file_b.py
  change_categories:
    - import_path_update
    - function_signature_update
    - config_update
    - test_update
  fallback_compatibility_required: true
```

### Step D. Patch the product

The agent should generate a patch that updates the product codebase.

### Step E. Run visible validation

The harness should run:

- install/build checks,
- visible tests,
- lints/type checks if included,
- product smoke test.

### Step F. Optional one repair loop

Allow one more edit iteration if visible tests fail.

### Step G. Run hidden evaluation

The harness runs hidden tests to determine final score.

---

## 7. Shared evaluation framework

For every episode, compute these metric families.

### 7.1 Product correctness

- hidden tests passed
- visible tests passed
- patch applied
- build/install success

### 7.2 Memory correctness

- vector DB mutation precision
- vector DB mutation recall
- vector DB mutation F1
- fresh-chunk retrieval rate
- stale-chunk retrieval rate
- whether expected new docs were cited

### 7.3 Adaptation efficiency

- files changed
- lines added
- lines removed
- patch attempt count
- retrieval query count
- docs fetched count
- reindex latency

### 7.4 Longitudinal metrics

Across the entire timeline of a product, measure:

- cumulative success rate,
- mean adaptation gain,
- regression rate,
- area under adaptation curve,
- cost-adjusted success,
- memory drift rate,
- stale retrieval persistence.

---

## 8. Shared test design: visible vs hidden

Each episode should have both visible and hidden tests.

### Visible tests

Purpose:

- provide actionable debugging signal,
- support one repair loop,
- ensure the task is not blind.

Examples:

- import/load smoke test,
- build or startup test,
- simple functional test,
- schema output test.

### Hidden tests

Purpose:

- prevent overfitting to visible surface behavior,
- test backward compatibility and edge cases,
- verify correct use of updated dependency semantics.

Examples:

- edge-case behavior,
- integration behavior,
- deprecation-safe compatibility,
- serialization contract,
- async scheduling behavior,
- endpoint schema,
- legacy fallback behavior.

### Rule

The agent must not directly access hidden tests, gold patches, or gold mutation manifests.

---

# 9. Project 1: Scrapy benchmark product

## 9.1 Product definition

Create a product named `crawler_app`.

Its stable business goal:

> crawl a set of fixture sites, extract items into a canonical JSON schema, support configurable crawling policies, and export a deterministic feed.

The product should evolve over time while preserving the same high-level outcome.

## 9.2 Product state progression

Use 5 states.

### S0 — baseline crawler

Capabilities:

- synchronous spider startup,
- crawl local HTML fixtures,
- extract item fields: `url`, `title`, `body`, `timestamp`,
- export JSONL feed,
- deterministic crawl order.

Dependencies:

- older stable Scrapy state,
- simple middleware pipeline.

### S1 — upgraded reactor / startup behavior

New dependency drift:

- Scrapy upgrade introduces asyncio reactor default and startup behavior changes.

New product requirement:

- preserve the same crawl output under the new execution model.

### S2 — async spider support

New dependency drift:

- `start_requests()` deprecation, async `start()` support, async iteration behavior.

New product requirement:

- support async startup while keeping stable feed output.

### S3 — middleware / policy evolution

New dependency drift:

- spider middleware async-output expectations,
- `allow_offsite` request metadata,
- updated request policy behavior.

New product requirement:

- selectively allow offsite requests for approved domains,
- middleware remains correct under async output.

### S4 — deprecation cleanup / extension refactor

New dependency drift:

- deprecated factory and import patterns are removed or discouraged.

New product requirement:

- replace deprecated extension or pipeline setup patterns,
- remove deprecated imports,
- preserve output schema.

## 9.3 Scrapy breakpoints

### Breakpoint B0: S0 → S1

Trigger:

- lockfile upgrades Scrapy to a version where the asyncio reactor is default.

Expected failure modes:

- reactor mismatch,
- startup warnings,
- test flakiness,
- subtle output order changes.

Agent must do:

- read release notes around reactor default,
- update vector DB,
- patch crawler settings,
- ensure deterministic scheduling/output remains stable.

#### Episode E1

Task:

- patch project settings for explicit reactor compatibility.

Visible tests:

- crawler boots,
- feed file generated.

Hidden tests:

- no duplicate items,
- deterministic output order,
- no reactor warnings in logs.

#### Episode E2

Task:

- patch scheduler/crawl startup behavior to preserve order under new reactor.

Visible tests:

- crawl returns expected item count.

Hidden tests:

- item ordering invariant,
- no missed seed URLs.

### Breakpoint B1: S1 → S2

Trigger:

- docs indicate `start_requests()` deprecation and async `start()` migration.

Expected failure modes:

- spider startup no longer uses intended entrypoint,
- warnings or missing requests,
- broken seed behavior.

Agent must do:

- read migration docs,
- update vector DB,
- refactor spider entrypoint from sync to async-compatible implementation,
- maintain compatibility strategy if benchmark requires legacy fallback.

#### Episode E3

Task:

- migrate spider startup to `start()`.

Visible tests:

- spider can crawl fixture site.

Hidden tests:

- seed URLs are identical to previous state,
- item set unchanged,
- no deprecated startup path used.

#### Episode E4

Task:

- update spider middleware to work with async startup flow.

Visible tests:

- middleware-enabled crawl succeeds.

Hidden tests:

- middleware preserves request annotations,
- no dropped requests/items.

### Breakpoint B2: S2 → S3

Trigger:

- docs introduce or emphasize `allow_offsite` and async spider-output constraints.

Expected failure modes:

- offsite filtering too strict or too permissive,
- async middleware emits invalid outputs.

Agent must do:

- read new request-meta and middleware docs,
- patch request generation and middleware behavior,
- preserve safety policy.

#### Episode E5

Task:

- add support for approved offsite crawl exceptions.

Visible tests:

- approved domain request succeeds.

Hidden tests:

- non-approved offsite requests remain blocked,
- feed still contains only allowed domains.

#### Episode E6

Task:

- adapt middleware output for async-compatible flow.

Visible tests:

- middleware test passes.

Hidden tests:

- transformed items match expected schema,
- no duplicate item processing.

### Breakpoint B3: S3 → S4

Trigger:

- deprecated factories/imports removed or discouraged.

Expected failure modes:

- extension initialization fails,
- import paths break,
- warnings promoted to failure.

Agent must do:

- replace deprecated construction methods,
- replace deprecated imports,
- update tests and initialization wiring.

#### Episode E7

Task:

- replace deprecated pipeline/extension factory pattern.

Visible tests:

- extension loads.

Hidden tests:

- extension lifecycle hooks run correctly.

#### Episode E8

Task:

- replace deprecated URL/helper imports and update affected code.

Visible tests:

- import smoke test.

Hidden tests:

- URL canonicalization behavior unchanged,
- no deprecated imports remain.

## 9.4 Scrapy files likely to change

Typical impacted files:

```text
crawler_app/spiders/*.py
crawler_app/settings.py
crawler_app/middlewares.py
crawler_app/pipelines.py
crawler_app/extensions.py
crawler_app/tests/*
```

## 9.5 Scrapy documentation registry

Store docs per state transition:

```text
registry/docs/scrapy/S1/
registry/docs/scrapy/S2/
registry/docs/scrapy/S3/
registry/docs/scrapy/S4/
```

Each should include:

- release note excerpt,
- migration note excerpt,
- deprecation excerpt,
- relevant API page excerpt.

---

# 10. Project 2: Prefect benchmark product

## 10.1 Product definition

Create a product named `workflow_app`.

Its stable business goal:

> execute a reproducible ETL / document-processing workflow with retries, parameter validation, logging, optional deployment, and optional integration-backed execution.

## 10.2 Product state progression

Use 5 states.

### P0 — local workflow baseline

Capabilities:

- local flow execution,
- task retries,
- artifact generation,
- parameterized input,
- deterministic local run summary.

### P1 — deployment / worker state

New product requirement:

- runnable via deployment and worker-compatible entrypoint.

### P2 — Prefect 3 migration

New dependency drift:

- Prefect 2.x → 3.x,
- Pydantic v2 relevance,
- module location/name changes,
- flow final-state semantics change.

New product requirement:

- keep workflow behavior and parameterization stable under Prefect 3.

### P3 — async/concurrency evolution

New dependency drift:

- async tasks in synchronous flows are no longer supported the same way.

New product requirement:

- support async-compatible execution.

### P4 — integration-backed workflow

New dependency drift:

- new integration package or updated integration release notes.

New product requirement:

- support one external integration-backed step with local fallback.

## 10.3 Prefect breakpoints

### Breakpoint B0: P0 → P1

Trigger:

- product requirement shifts from local-only flow to deployable flow.

Expected failure modes:

- missing deployment metadata,
- worker startup mismatch,
- packaging problems.

Agent must do:

- read deployment/worker docs,
- update vector DB,
- add deployment configuration,
- make workflow executable in deployment mode.

#### Episode E1

Task:

- introduce deployable entrypoint / deployment config.

Visible tests:

- local flow still runs.

Hidden tests:

- deployment config is valid,
- worker-compatible run bootstrap succeeds.

#### Episode E2

Task:

- patch CI to support both local and deployment smoke runs.

Visible tests:

- local CI green.

Hidden tests:

- deployment command path works without modifying hidden infra assumptions.

### Breakpoint B1: P1 → P2

Trigger:

- Prefect upgrade to 3.x.

Expected failure modes:

- parameter model mismatch due to Pydantic v2,
- import path warnings/errors,
- changed flow final-state assertions.

Agent must do:

- read upgrade guide,
- ingest module path changes and Pydantic notes,
- update parameter models,
- patch import paths,
- revise state assertions.

#### Episode E3

Task:

- migrate custom parameter models or blocks for Pydantic v2 compatibility.

Visible tests:

- parameterized flow validates.

Hidden tests:

- edge-case parameter schemas pass,
- no direct Prefect-object incompatibility remains.

#### Episode E4

Task:

- update final-state logic and affected tests after Prefect 3 semantics change.

Visible tests:

- flow run completes.

Hidden tests:

- failing task states map to correct flow final state,
- previous assertions updated correctly.

### Breakpoint B2: P2 → P3

Trigger:

- Prefect docs note async tasks in synchronous flows are no longer supported the same way.

Expected failure modes:

- runtime errors,
- blocked futures,
- incorrect flow/task behavior.

Agent must do:

- detect sync/async mismatch,
- refactor flow or task runner,
- preserve business logic.

#### Episode E5

Task:

- make the flow async-compatible.

Visible tests:

- core ETL path runs.

Hidden tests:

- async branch executes correctly,
- retries still behave as expected,
- flow state is correct.

#### Episode E6

Task:

- update concurrency or task-runner settings to support the refactor.

Visible tests:

- concurrent tasks execute.

Hidden tests:

- no deadlocks,
- deterministic output persists.

### Breakpoint B3: P3 → P4

Trigger:

- integration package added or upgraded.

Expected failure modes:

- missing imports,
- outdated integration config,
- credentials/fallback assumptions.

Agent must do:

- read integration-specific release notes,
- patch workflow to use new integration API,
- preserve local fallback path for benchmark determinism.

#### Episode E7

Task:

- add one integration-backed step (e.g. Docker/GitHub/Redis/local storage equivalent).

Visible tests:

- fallback local path passes.

Hidden tests:

- integration path is syntactically valid and correctly wired.

#### Episode E8

Task:

- patch config/schema for upgraded integration package.

Visible tests:

- config loads.

Hidden tests:

- integration settings validate,
- workflow still runs without network credentials by using fallback path.

## 10.4 Prefect files likely to change

Typical impacted files:

```text
workflow_app/flows/*.py
workflow_app/tasks/*.py
workflow_app/models.py
workflow_app/deployments/*.yaml
workflow_app/integrations/*.py
workflow_app/tests/*
```

## 10.5 Prefect documentation registry

Store docs per breakpoint:

```text
registry/docs/prefect/P1/
registry/docs/prefect/P2/
registry/docs/prefect/P3/
registry/docs/prefect/P4/
```

Include:

- upgrade guide excerpts,
- OSS release note excerpts,
- integration release note excerpts,
- example code snippets.

---

# 11. Project 3: Haystack benchmark product

## 11.1 Product definition

Create a product named `rag_app`.

Its stable business goal:

> ingest documents, answer questions over them, support configurable retrieval strategies, and expose the QA system through a service interface.

## 11.2 Product state progression

Use 5 states.

### H0 — basic extractive QA baseline

Capabilities:

- local document ingestion,
- basic retrieval,
- extractive QA or minimal answer generation,
- CLI or notebook-style local interface.

### H1 — Haystack 2 migration

New dependency drift:

- package rename `farm-haystack` → `haystack-ai`,
- nodes → components,
- pipeline rewrite.

New product requirement:

- preserve document ingestion and QA behavior under Haystack 2.x.

### H2 — generative RAG expansion

New dependency drift:

- prompt/generator patterns evolve,
- `PromptNode`-style legacy pattern replaced by prompt builder + generator composition.

New product requirement:

- support generative answer formatting.

### H3 — retriever / document store evolution

New dependency drift:

- retriever and document-store APIs evolve.

New product requirement:

- support selectable retrieval mode while preserving answer quality.

### H4 — service deployment evolution

New dependency drift:

- older REST deployment pattern replaced by Hayhooks-style deployment.

New product requirement:

- expose the RAG product as a service endpoint.

## 11.3 Haystack breakpoints

### Breakpoint B0: H0 → H1

Trigger:

- major migration guide says uninstall `farm-haystack`, install `haystack-ai`; pipeline architecture changes.

Expected failure modes:

- imports fail,
- old nodes absent,
- pipelines cannot be constructed.

Agent must do:

- read migration guide,
- update vector DB,
- rewrite imports,
- refactor pipeline construction,
- preserve product API.

#### Episode E1

Task:

- migrate indexing pipeline to Haystack 2-style components.

Visible tests:

- documents can be indexed.

Hidden tests:

- indexed documents retrievable,
- no legacy imports remain,
- product abstraction preserved.

#### Episode E2

Task:

- migrate query pipeline to Haystack 2-style pipeline execution.

Visible tests:

- a simple QA query returns an answer.

Hidden tests:

- retrieval + answer path both operate,
- query schema preserved.

### Breakpoint B1: H1 → H2

Trigger:

- docs indicate generator + `PromptBuilder` patterns replacing earlier prompt/generation composition.

Expected failure modes:

- prompt wiring broken,
- answer parser broken,
- generator not called correctly.

Agent must do:

- ingest migration docs,
- patch prompt building and generator connection,
- preserve answer schema.

#### Episode E3

Task:

- replace legacy prompt-generation logic with prompt builder + generator.

Visible tests:

- prompt-generation path runs.

Hidden tests:

- retrieved context appears in prompt,
- answer field schema preserved,
- deterministic test prompt passes.

#### Episode E4

Task:

- patch answer parser / response postprocessor for new generator output format.

Visible tests:

- answer JSON validates.

Hidden tests:

- citations/context fields remain correct,
- hallucination guardrails still pass fixed heuristics.

### Breakpoint B2: H2 → H3

Trigger:

- retriever and document-store choices evolve.

Expected failure modes:

- indexing broken,
- retrieval backend config mismatch,
- query path silently degrades.

Agent must do:

- read retriever/document-store docs,
- patch retrieval backend abstraction,
- optionally maintain fallback retrieval mode.

#### Episode E5

Task:

- add configurable retriever backend.

Visible tests:

- BM25 baseline still works.

Hidden tests:

- alternate retriever mode works,
- retrieval quality threshold met on fixture dataset.

#### Episode E6

Task:

- update ingestion path for changed document-store interface.

Visible tests:

- ingestion completes.

Hidden tests:

- document IDs stable,
- duplicate ingestion avoided,
- retrieval freshness preserved.

### Breakpoint B3: H3 → H4

Trigger:

- deployment approach shifts to Hayhooks-style serving.

Expected failure modes:

- old REST deployment assumptions invalid,
- serving contract broken,
- endpoint serialization mismatched.

Agent must do:

- ingest deployment docs,
- patch service interface,
- update endpoint tests,
- preserve product query contract.

#### Episode E7

Task:

- add service wrapper for the RAG app using the current deployment approach.

Visible tests:

- local service smoke test passes.

Hidden tests:

- endpoint schema valid,
- pipeline called correctly behind endpoint.

#### Episode E8

Task:

- patch deployment config / request handling for new serving model.

Visible tests:

- request-response smoke test.

Hidden tests:

- malformed input handled correctly,
- answer schema and citation structure preserved.

## 11.4 Haystack files likely to change

Typical impacted files:

```text
rag_app/pipeline/*.py
rag_app/indexing/*.py
rag_app/retrieval/*.py
rag_app/service/*.py
rag_app/config/*.yaml
rag_app/tests/*
```

## 11.5 Haystack documentation registry

Store docs per breakpoint:

```text
registry/docs/haystack/H1/
registry/docs/haystack/H2/
registry/docs/haystack/H3/
registry/docs/haystack/H4/
```

Include:

- migration guide excerpts,
- package rename notes,
- prompt/generator migration examples,
- retriever/document-store examples,
- Hayhooks deployment excerpts.

---

# 12. Shared vector DB update protocol

For every project and every transition, use the same vector DB protocol.

## 12.1 Chunking rules

- chunk deterministically,
- stable `chunk_id = hash(document_path + chunk_index + normalized_text)`,
- attach metadata:
  - project,
  - state,
  - dependency,
  - doc type,
  - version,
  - available_at.

## 12.2 Mutation operations

At each transition, compare newly visible docs against prior docs and emit:

- `insert`
- `update`
- `delete`
- `tombstone`

## 12.3 Required benchmark artifacts

For every breakpoint:

```text
artifacts/
  gold_mutations.yaml
  expected_retrieval.yaml
  freshness_labels.yaml
```

## 12.4 What the agent must output

At minimum:

```json
{
  "ingested_documents": [...],
  "mutation_records": [...],
  "retrieval_queries": [...],
  "retrieved_chunk_ids": [...],
  "cited_chunk_ids": [...]
}
```

---

# 13. Environment freeze design

Use **Docker + Conda/uv lockfiles + local wheelhouse**.

## 13.1 Rule

Never rely on live internet package installation in official evaluation.

Each state should have:

```text
envs/<project>/<state_id>/
  Dockerfile
  environment.yml
  conda-lock.yml
  requirements.lock
  wheelhouse/
```

## 13.2 Build hierarchy

Use layered images:

```text
base image → project base image → state image
```

Example:

```text
clab/base:py311
clab/scrapy:base
clab/scrapy:S0
clab/scrapy:S1
...
```

## 13.3 State validation

Before making an episode official, validate that the state image can:

- build/install product,
- run visible tests,
- run hidden tests using gold reference code,
- serialize environment metadata.

---

# 14. Dataset spec to implement

Every project should have:

```text
timelines/<project>.yaml
states/<project>/<state_id>/state.yaml
episodes/<project>/<episode_id>.yaml
```

## 14.1 Timeline manifest

Example:

```yaml
project: scrapy
product_name: crawler_app
states:
  - S0
  - S1
  - S2
  - S3
  - S4
transitions:
  - from: S0
    to: S1
    breakpoint: B0
    episodes: [E1, E2]
  - from: S1
    to: S2
    breakpoint: B1
    episodes: [E3, E4]
  - from: S2
    to: S3
    breakpoint: B2
    episodes: [E5, E6]
  - from: S3
    to: S4
    breakpoint: B3
    episodes: [E7, E8]
```

## 14.2 Episode manifest

Example:

```yaml
episode_id: scrapy_E3
project: scrapy
product: crawler_app
from_state: S1
to_state: S2
breakpoint: B1
breakpoint_type:
  - async_sync_semantics_change
  - deprecation_removal
visible_docs:
  - registry/docs/scrapy/S2/release_notes_excerpt.md
  - registry/docs/scrapy/S2/migration_excerpt.md
visible_failure_signal:
  - states/scrapy/S2/tests/visible/failure_log.txt
task_prompt: |
  The crawler has been upgraded to a newer Scrapy state.
  Startup behavior has changed and the existing spider entrypoint is no longer valid.
  Read the new docs, update the vector DB, and patch the crawler to preserve the expected output.
expected_artifacts:
  - mutation_log.jsonl
  - retrieval_trace.jsonl
  - patch.diff
success_criteria:
  hidden_tests_passed: true
  visible_tests_passed: true
  patch_applied: true
```

---

# 15. Memory persistence policy

Support two modes.

## 15.1 Persistent mode (main benchmark)

The vector DB persists across states.

```text
DB_S0 → DB_S1 → DB_S2 → DB_S3 → DB_S4
```

This is the main continual-learning setting.

## 15.2 Reset mode (baseline)

The vector DB resets at every episode.

Use this as a baseline only.

## 15.3 Required evaluation comparison

For each project, compare at least:

1. no retrieval,
2. stale DB only,
3. updated DB persistent mode,
4. oracle retrieval.

This is necessary to show that external-memory updates actually matter.

---

# 16. Recommended implementation order for the coding agent

## Phase 1 — framework scaffold

Implement:

- timeline schema,
- state schema,
- episode schema,
- mutation log schema,
- retrieval trace schema,
- Docker/lockfile runner,
- visible/hidden test runner.

## Phase 2 — Scrapy product first

Build the full `crawler_app` benchmark first.

Reason:

- easiest state freezing,
- deterministic local fixtures,
- low dependency cost,
- simplest hidden tests.

Deliverables:

- S0–S4,
- E1–E8,
- working persistent vector DB evaluation,
- reports.

## Phase 3 — Prefect product second

Build `workflow_app`.

Deliverables:

- P0–P4,
- E1–E8,
- deployment/integration stubs,
- async/sync validation.

## Phase 4 — Haystack product third

Build `rag_app`.

Deliverables:

- H0–H4,
- E1–E8,
- retrieval/document-store abstractions,
- service endpoint evaluation.

## Phase 5 — unify reporting

Create cross-project reporting:

- per-episode report,
- per-project longitudinal report,
- cross-project leaderboard.

---

# 17. What “done” means for v0

The benchmark is ready when all of the following are true:

## Dataset completeness

- 3 products exist,
- each has 5 states,
- each has 8 episodes,
- all state images build reproducibly,
- all visible docs are timestamped and stored in registry.

## Agent integration completeness

- coding agent can ingest docs,
- coding agent can mutate vector DB,
- coding agent can generate patch,
- framework logs mutations and retrievals,
- framework evaluates visible and hidden tests.

## Scientific completeness

- persistent-memory mode works,
- reset baseline works,
- stale DB baseline works,
- oracle baseline works,
- per-state and longitudinal metrics are emitted.

## Documentation completeness

- every state and breakpoint documented,
- every episode has task prompt and success criteria,
- build instructions are reproducible,
- leakage rules are explicit.

---

# 18. Final concrete next actions for the coding agent

1. **Create the monorepo scaffold.**
2. **Implement the generic timeline/state/episode schemas.**
3. **Build the Scrapy benchmark product first.**
4. **Freeze Scrapy states S0–S4 with local HTML fixtures.**
5. **Write E1–E8 episodes for Scrapy.**
6. **Implement vector DB mutation logging and retrieval traces.**
7. **Implement visible/hidden test execution.**
8. **Run the first persistent-memory baseline on Scrapy.**
9. **Only after Scrapy works, add Prefect states and episodes.**
10. **Only after Prefect works, add Haystack states and episodes.**
11. **Add cross-project scoring and longitudinal reports.**
12. **Publish a benchmark spec and reproducibility guide.**

---

# 19. Reference links used for the current design

## Scrapy

- PyPI project metadata and license: https://pypi.org/project/Scrapy/
- Scrapy release notes: https://docs.scrapy.org/en/master/news.html
- Scrapy versioning and API stability: https://docs.scrapy.org/en/2.13/versioning.html

## Prefect

- Prefect GitHub repo and license: https://github.com/PrefectHQ/prefect
- Prefect 3.0 upgrade guide: https://docs.prefect.io/latest/guides/migration-guide
- Prefect release notes: https://docs.prefect.io/v3/release-notes

## Haystack

- Haystack GitHub repo: https://github.com/deepset-ai/haystack
- Haystack license file: https://github.com/deepset-ai/haystack/blob/main/LICENSE
- Haystack 2.0 migration guide: https://docs.haystack.deepset.ai/v2.0/docs/migration

## SWE-bench inspiration

- User-provided SWE-bench paper PDF in current workspace.

---

# 20. One-sentence benchmark definition

> This benchmark evaluates whether a coding agent can survive product-line evolution across a simulated timeline by reading newly available dependency documentation, updating its vector database, patching the product code, and maintaining correct behavior under hidden CI-style evaluation across multiple evolving software states.
