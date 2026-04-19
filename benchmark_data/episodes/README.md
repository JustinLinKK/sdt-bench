# Episodes

Each episode directory is self-contained and includes:

- `episode.yaml` - validated episode spec
- `visible_docs/` - only new external docs that may be ingested
- `hidden_eval/` - harness-only tests and manifests
- `artifacts/` - scoring artifacts such as expected mutations or retrieval chunks

The bundled v0 example is `benchmark_data/episodes/requests/episode_0001`.
