install:
	uv sync --dev

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest

smoke:
	uv run pytest tests/test_smoke_integration.py

