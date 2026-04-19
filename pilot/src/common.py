"""Shared utilities: paths, config loader, OpenRouter client, chunk hashing."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CONFIG = ROOT / "config" / "repos.yaml"

PROJECT_ROOT = ROOT.parents[1]


def load_env() -> None:
    for candidate in (PROJECT_ROOT / ".env.macos", PROJECT_ROOT / ".env", ROOT / ".env"):
        if candidate.exists():
            load_dotenv(candidate, override=False)


def load_config() -> dict[str, Any]:
    with CONFIG.open() as f:
        return yaml.safe_load(f)


def chunk_id(path: str, content: str) -> str:
    h = hashlib.sha256()
    h.update(path.encode())
    h.update(b"\0")
    h.update(content.encode())
    return h.hexdigest()[:16]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


@dataclass
class OpenRouterClient:
    api_key: str
    base_url: str = "https://openrouter.ai/api/v1"

    @classmethod
    def from_env(cls) -> "OpenRouterClient":
        load_env()
        key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not key or key == "YOUR_OPENROUTER_KEY_HERE":
            raise RuntimeError("OPENROUTER_API_KEY is not set. Populate .env.macos.")
        return cls(api_key=key)

    def chat(self, model: str, messages: list[dict], **kwargs) -> dict:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        resp = client.chat.completions.create(model=model, messages=messages, **kwargs)
        return resp.model_dump()

    def embed(self, model: str, inputs: list[str]) -> list[list[float]]:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        resp = client.embeddings.create(model=model, input=inputs)
        return [d.embedding for d in resp.data]
