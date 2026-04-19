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
sdt-bench author-harvest-releases --repo-name requests
```

## Design rules

- keep state explicit and serializable
- avoid hidden global state
- prefer deterministic IDs, timestamps, and artifact naming
- keep benchmark logic generic and repo-specific behavior isolated
- keep bundled baselines separate from the benchmark core package
