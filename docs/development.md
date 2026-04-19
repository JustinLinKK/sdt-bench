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
```

## Design rules

- keep state explicit and serializable
- avoid hidden global state
- prefer deterministic IDs, timestamps, and artifact naming
- keep benchmark logic generic and repo-specific behavior isolated

