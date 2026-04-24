# Projects

`benchmark_data/projects/` is the canonical benchmark dataset root.

Each project owns its full benchmark timeline:

- `project.yaml` for project metadata
- `timeline.yaml` for the ordered state and episode sequence
- `states/` for immutable product snapshots, visible docs, and visible tests
- `events/` for dependency-drift transitions and event-scoped visible docs
- `episodes/` for agent-facing tasks, failure signals, and hidden evaluation

Current bundled projects:

- `toy`
- `crawler_app`
- `workflow_app`
- `rag_app`
