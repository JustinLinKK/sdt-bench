FROM python:3.11-slim

WORKDIR /app

ENV UV_LINK_MODE=copy

RUN pip install --no-cache-dir uv

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY configs ./configs
COPY data ./data
COPY docs ./docs

RUN uv sync --no-dev

ENTRYPOINT ["uv", "run", "python", "-m", "sdt_bench.cli"]
