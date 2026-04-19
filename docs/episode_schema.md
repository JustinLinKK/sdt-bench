# Episode Schema

## RepoSpec

- `name`: benchmark repo identifier
- `github_url`: upstream source URL or clone target
- `default_branch`: default branch name
- `language`: implementation language
- `package_manager`: install tool used in the workspace
- `install_command`: shell command used during materialization
- `test_command`: default visible test command
- `dockerfile_path`: optional Dockerfile path for reproducible execution
- `image_reference`: optional prebuilt image reference
- `package_name`: package index name used by authoring commands
- `ecosystem`: package ecosystem used by authoring commands
- `supported_dependency_files`: dependency manifests relevant to the repo
- `notes`: free-form implementation notes

## DependencyEvent

- `event_id`: stable identifier for the drift event
- `dependency_name`: upgraded dependency
- `ecosystem`: package ecosystem such as `pypi`
- `old_version`: previous dependency version
- `new_version`: upgraded dependency version
- `event_type`: one of `patch`, `minor`, `major`, `security`, `synthetic`
- `summary`: human-readable description
- `breaking_change_expected`: whether breakage is expected
- `visible_doc_paths`: visible docs associated with the event
- `gold_mutation_paths`: mutation artifacts used for scoring
- `metadata`: free-form event metadata

## EpisodeSpec

- `episode_id`: stable episode identifier
- `repo_name`: repo config key
- `base_commit`: pinned commit used for materialization
- `base_ref`: human-readable ref or tag
- `dependency_event`: embedded `DependencyEvent`
- `task_family`: task category
- `task_prompt`: prompt shown to the agent
- `visible_failure_signal`: visible failure description
- `visible_doc_paths`: visible doc paths relative to the episode directory
- `hidden_test_command`: command template used for hidden tests
- `hidden_test_manifest`: manifest path relative to the episode directory
- `gold_patch_path`: optional gold patch
- `expected_files_of_interest`: files likely relevant to the task
- `success_criteria`: per-episode evaluation thresholds
- `budget`: execution budget hints
  - includes `acquisition_budget` for agent-controlled visible-doc acquisition
- `notes`: free-form notes
