# Dataset Schemas

## `TemporalStateSpec`

- `state_id`
- `timeline_id`
- `project_id`
- `timestamp`
- `source_ref`
- `dependency_snapshot`
- `environment.install_command`
- `environment.offline`
- `environment.visible_test_command`
- `snapshot_root`
- `docs_manifest_path`

## `DependencyEventSpec`

- `event_id`
- `from_state_id`
- `to_state_id`
- `dependency_name`
- `old_version`
- `new_version`
- `event_type`
- `summary`
- `available_at`
- `visible_doc_paths`
- `gold_mutation_paths`
- `expected_retrieval_path`

## `ProgrammingEpisodeSpec`

- `episode_id`
- `timeline_id`
- `project_id`
- `from_state_id`
- `to_state_id`
- `event_id`
- `task_family`
- `task_prompt`
- `visible_failure_path`
- `hidden_test_command`
- `hidden_test_manifest`
- `budget`
- `success_criteria`

## `TimelineSpec`

- `timeline_id`
- `project_id`
- `initial_state_id`
- `state_ids`
- `episode_ids`
- `default_memory_mode`
