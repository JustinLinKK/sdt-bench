# Development

## Tooling

- Python 3.11 target
- `uv` for dependency management
- `ruff` for linting and formatting checks
- `pytest` for unit and smoke tests

## Useful commands

```bash
make install
make lint
make test
make smoke
sdt-bench author-harvest-releases --project-id crawler_app
```

## Design rules

- keep state explicit and serializable
- avoid hidden global state
- prefer deterministic IDs, timestamps, and artifact naming
- keep benchmark logic generic and project-specific behavior isolated
- keep bundled baselines separate from the benchmark core package
