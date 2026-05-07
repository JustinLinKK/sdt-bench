"""Claude SDK backend for MLEvolve.

Uses claude_agent_sdk.query() which spawns the `claude` CLI subprocess for each
call. The CLI is already authenticated via Pro/Max subscription, so no API key
is required. We intentionally pass setting_sources=[], skills=[], tools=[] to
ensure each call is a clean assistant interaction without local skills or
session context (e.g. caveman skill) leaking into MLEvolve generations.

Mirrors `llm/openai.py` shape: `query()` returns
(output, req_time, in_tok, out_tok, info); `generate()` returns string.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any

from claude_agent_sdk import query as _sdk_query, ClaudeAgentOptions

from config import Config
from .gemini import FunctionSpec, OutputType, compile_prompt_to_md
from .openai import _parse_json_args, _extract_json_object

logger = logging.getLogger("MLEvolve")


_DEFAULT_MODEL = "claude-opus-4-7"


def _strip_thinking(text: str) -> str:
    """Remove any <thinking>...</thinking> blocks. Claude opus models can emit
    these when extended thinking is on; for code generation we want only the
    final answer.
    """
    return re.sub(r"<thinking>.*?</thinking>\s*", "", text or "", flags=re.DOTALL)


def _build_options(model: str, system_prompt: str | None) -> ClaudeAgentOptions:
    """Build clean ClaudeAgentOptions: no skills, no tools, no project settings,
    so each call is a stateless prompt-response with the chosen model."""
    return ClaudeAgentOptions(
        model=model or _DEFAULT_MODEL,
        system_prompt=(system_prompt or
                       "You are a helpful coding assistant. Reply with the requested content only."),
        setting_sources=[],   # do not load user/project/local settings
        skills=[],             # no skills
        tools=[],              # no tool use; pure text generation
        max_turns=1,           # single response per call
    )


async def _run_query_async(prompt: str, model: str, system_prompt: str | None) -> tuple[str, dict]:
    """Send single prompt to Claude SDK, collect text + token usage."""
    options = _build_options(model, system_prompt)
    text = ""
    info: dict[str, Any] = {"model": model}
    in_tok = 0
    out_tok = 0
    async for msg in _sdk_query(prompt=prompt, options=options):
        # Accumulate text from AssistantMessage content blocks
        for block in getattr(msg, "content", []) or []:
            if hasattr(block, "text"):
                text += block.text
        # ResultMessage carries usage / cost info
        if hasattr(msg, "usage") and msg.usage is not None:
            try:
                in_tok = int(getattr(msg.usage, "input_tokens", 0) or 0)
                out_tok = int(getattr(msg.usage, "output_tokens", 0) or 0)
            except Exception:
                pass
        if hasattr(msg, "session_id"):
            info["session_id"] = msg.session_id
        if hasattr(msg, "total_cost_usd") and msg.total_cost_usd is not None:
            try:
                info["total_cost_usd"] = float(msg.total_cost_usd)
            except Exception:
                pass
    return text, {"in_tok": in_tok, "out_tok": out_tok, **info}


def _run_query(prompt: str, model: str, system_prompt: str | None) -> tuple[str, dict]:
    """Sync wrapper around async query. Each call gets its own event loop."""
    return asyncio.run(_run_query_async(prompt, model, system_prompt))


def _build_chat_text(system_message: str | None,
                     user_message: str | None,
                     prompt_dict: dict | None = None) -> tuple[str, str | None]:
    """Combine system + user into a single user prompt for Claude SDK.

    Claude SDK supports system_prompt as a separate field; the user prompt is
    a single string. If MLEvolve passes a chat-style dict (system/user/assistant)
    we collapse it to (system_prompt, user_prompt) form.
    """
    if prompt_dict is not None:
        sys_msg = str(prompt_dict.get("system", "")) if prompt_dict.get("system") else None
        user_msg = str(prompt_dict.get("user", "")) if prompt_dict.get("user") else ""
        assistant = str(prompt_dict.get("assistant", "")) if prompt_dict.get("assistant") else ""
        if assistant:
            user_msg = f"{user_msg}\n\n{assistant}" if user_msg else assistant
        return user_msg or "(empty)", sys_msg
    return (user_message or ""), system_message


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    cfg: Config | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    """Claude SDK query with optional FunctionSpec.

    If func_spec is provided, we encode the JSON schema in the system prompt
    and ask the model to reply with a single JSON object. We parse it back to
    a dict (compatible with FunctionSpec consumers).
    """
    filtered = {k: v for k, v in model_kwargs.items() if v is not None}
    model = filtered.get("model", _DEFAULT_MODEL)
    user_text, sys_text = _build_chat_text(system_message, user_message)

    if func_spec is not None:
        # Inject function-call schema into system prompt and ask for JSON object reply.
        schema_str = json.dumps(func_spec.json_schema, indent=2) if hasattr(func_spec, "json_schema") else "{}"
        sys_text = ((sys_text or "") + "\n\n" if sys_text else "") + (
            f"You must reply with a single JSON object satisfying this schema (no prose):\n{schema_str}"
        )

    t0 = time.time()
    text, info = _run_query(user_text, model, sys_text)
    req_time = time.time() - t0
    text = _strip_thinking(text).strip()

    if func_spec is None:
        output = text
    else:
        # Parse JSON output back to dict
        try:
            output = _parse_json_args(text)
        except json.JSONDecodeError:
            output = _parse_json_args(_extract_json_object(text))
    in_tok = info.get("in_tok", 0)
    out_tok = info.get("out_tok", 0)
    return output, req_time, in_tok, out_tok, {"model": model, **info}


def _prompt_to_text(prompt: str | dict | list, model: str = "") -> tuple[str, str | None]:
    """Convert MLEvolve-style prompt (str / chat dict / list) to (user_text, system_text)."""
    if isinstance(prompt, dict) and ("system" in prompt or "user" in prompt or "assistant" in prompt):
        return _build_chat_text(None, None, prompt_dict=prompt)
    if isinstance(prompt, str):
        return prompt, None
    # list or any other type — compile to markdown and treat as user
    text = compile_prompt_to_md(prompt)
    return text, None


def generate(
    prompt: str | dict | list,
    cfg: Config,
    temperature: float | None = None,
    max_tokens: int | None = None,
    stop_tokens: list[str] | None = None,
    json_schema: dict | None = None,
    max_retries: int = 5,
    retry_delay: float = 3.0,
) -> str:
    """Streaming text generation via Claude SDK.

    Note: temperature and stop_tokens are ignored (Claude SDK does not expose them).
    The SDK respects model defaults. json_schema is honored by injecting an
    instruction into the system prompt; the model reply is parsed if needed.
    """
    stage = cfg.agent.code
    model = getattr(stage, "model", _DEFAULT_MODEL) or _DEFAULT_MODEL
    user_text, sys_text = _prompt_to_text(prompt, model=model)

    if json_schema is not None:
        schema_str = json.dumps(json_schema, indent=2)
        sys_text = ((sys_text or "") + "\n\n" if sys_text else "") + (
            f"Reply with ONLY a single JSON object satisfying this schema (no prose):\n{schema_str}"
        )

    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            text, _info = _run_query(user_text, model, sys_text)
            text = _strip_thinking(text)
            logger.info(f"claude_sdk.generate response ({len(text)} chars)", extra={"verbose": True})
            return text
        except Exception as exc:
            last_exc = exc
            logger.warning(f"claude_sdk.generate failed, retrying {attempt + 1}/{max_retries}: {exc}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    raise RuntimeError(f"claude_sdk.generate failed after {max_retries} retries: {last_exc}")
