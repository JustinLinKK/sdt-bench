# Evaluation

## Patch application

The harness validates unified diffs with `git apply --check` before applying them.
No-op patches are treated as successfully applied if they are empty.

## Visible tests vs hidden tests

Visible tests are optional and typically reflect what the agent is allowed to know
about the failure mode. Hidden tests are always harness-owned and are executed from
the episode’s hidden evaluation command.

## Metrics

v0 computes:

- patch application status
- visible and hidden test pass/fail
- files changed, lines added, lines removed
- fresh and stale retrieval fractions
- required-fresh-chunk retrieval
- mutation precision, recall, and F1
- citation overlap
- final weighted score

If present, `mutation_log.jsonl` from the agent run is the source of truth for
mutation-quality scoring.

## Report generation

The `report` command converts structured artifacts into a markdown summary with
episode metadata, dependency event details, mutation stats, retrieval freshness,
patch statistics, test outcomes, and the final score.
