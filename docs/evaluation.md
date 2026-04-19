# Evaluation

## Step-level scoring

Each step computes:

- patch application status
- visible and hidden test status
- mutation precision, recall, and F1
- fresh and stale retrieval fractions
- citation overlap
- churn score
- final weighted score

## Timeline-level scoring

`run-timeline` also aggregates:

- hidden pass rate
- cumulative success
- adaptation area
- average stale retrieval rate
- mean time to recover
- max drawdown

## Output ownership

- Agents own `output/`
- The harness owns `harness/result.json`, `harness/report.md`, `harness/visible_tests.json`, and `harness/hidden_tests.json`
