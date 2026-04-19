# Dataset Schemas

## `TemporalStateSpec`

- `state_id`
- `timeline_id`
- `repo_name`
- `timestamp`
- `repo_commit`
- `dependency_snapshot`
- `environment.install_command`
- `environment.offline`
- `environment.visible_test_command`
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
- `repo_name`
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
- `repo_name`
- `initial_state_id`
- `state_ids`
- `episode_ids`
- `default_memory_mode`
