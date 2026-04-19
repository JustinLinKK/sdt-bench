# Benchmark Spec

## What an episode is

An episode is a replayable unit of benchmark evaluation. It binds together:

- a pinned repository snapshot
- a controlled dependency drift event
- visible docs that may be ingested into external memory
- hidden evaluation instructions and tests
- scoring artifacts and expectations

## What is visible to the agent

The agent can only observe:

- the task prompt
- the visible failure signal
- retrieved chunks from the visible-doc vector store
- the materialized repository workspace

## What is hidden

The following remain harness-only:

- hidden evaluation tests and manifests
- scoring logic details beyond published metrics
- mutation artifacts used for scoring unless explicitly exposed

## Dependency-drift event definition

A dependency-drift event captures the change from one version of a dependency to
another, along with supporting visible documents and metadata about expected risk.

## Success criteria

Success is measured with both raw and aggregate metrics:

- patch application status
- visible and hidden test outcomes
- mutation-log precision, recall, and F1
- retrieval freshness and stale retrieval rate
- citation overlap with expected relevant docs
- code churn and final weighted score

