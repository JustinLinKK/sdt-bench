# Project Onboarding

To add a new benchmark project:

1. create `benchmark_data/projects/<project_id>/project.yaml`
2. add `timeline.yaml`, `states/`, `events/`, and `episodes/` under that project root
3. place runnable source snapshots under each `states/<state_id>/product_snapshot/`
4. define visible docs, hidden evaluation, and scoring artifacts for each episode
5. add targeted tests if the project requires custom environment behavior

Keep project-specific logic inside `src/sdt_bench/projects/` so the generic benchmark modules
remain reusable.
