"""Trace recorder for MLEvolve Phase 1 capture.

Monkey-patches:
  - llm.generate / llm.query                — record per-LLM-call (phase, start, end, duration, tokens)
  - AgentSearch._run_single_step             — record per-step + which agent fired + child node id
  - AgentSearch.execute_deferred_node        — record Phase-2 deferred drafts
  - Interpreter.run                          — record exec_submit_at / exec_complete_at + per-try info
  - DebugAgent diff retry loop               — tag each LLM call with retry_idx

Output is one JSONL line per *submission* (not per node), so each debug retry
becomes its own submission entry sharing the parent node_id.

Output file: $TRACE_OUT (default experiments/sweep_real_delays_2026-05-06/workload_trace_real.jsonl)
Per-submission code copied to: $CODE_DIR (default …/replay_codes/sub_NNN.py)
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

REPO = "/home/downeyflyfan/Research_Projects/AI/Agents/Agents_Scheduler/MLEvolve_Schedule"
sys.path.insert(0, REPO)

EXP_DIR = Path(__file__).resolve().parent
TRACE_OUT = Path(os.environ.get("TRACE_OUT", EXP_DIR / "workload_trace_real.jsonl"))
CODE_DIR = Path(os.environ.get("CODE_DIR", EXP_DIR / "replay_codes"))
TRACE_OUT.parent.mkdir(parents=True, exist_ok=True)
CODE_DIR.mkdir(parents=True, exist_ok=True)

# Truncate trace + clear codes on fresh recorder load
if os.environ.get("TRACE_RESET", "1") == "1":
    if TRACE_OUT.exists():
        TRACE_OUT.unlink()
    for f in CODE_DIR.glob("sub_*.py"):
        f.unlink()

_lock = threading.RLock()
_t0 = None
_pending_step: dict | None = None
_step_idx_counter = 0
_submission_idx_counter = 0
_current_retry_idx = 0   # set by debug-agent wrapper; resets each step
_current_phase = "unknown"


def _now_rel() -> float:
    global _t0
    if _t0 is None:
        _t0 = time.time()
    return time.time() - _t0


def _flush_submission(entry: dict) -> None:
    with _lock:
        with TRACE_OUT.open("a") as f:
            f.write(json.dumps(entry, default=str) + "\n")


def _record_llm_call(purpose: str, t_start: float, t_end: float,
                     in_tok: int = 0, out_tok: int = 0, error: bool = False):
    """Append an LLM call record into the currently-pending step."""
    if _pending_step is None:
        return
    rec = {
        "phase": _current_phase if not error else f"{_current_phase}_error",
        "purpose": purpose,
        "start_at": round(t_start - _t0, 3) if _t0 else 0.0,
        "end_at": round(t_end - _t0, 3) if _t0 else 0.0,
        "duration_s": round(t_end - t_start, 3),
        "retry_idx": _current_retry_idx,
        "in_tok": int(in_tok or 0),
        "out_tok": int(out_tok or 0),
    }
    with _lock:
        _pending_step["llm_calls"].append(rec)


def install_patches() -> None:
    # 1) llm.generate + llm.query
    try:
        import llm
        original_generate = llm.generate

        def traced_generate(prompt, cfg, **kwargs):
            t_start = time.time()
            try:
                result = original_generate(prompt=prompt, cfg=cfg, **kwargs)
                _record_llm_call("generate", t_start, time.time())
                return result
            except Exception:
                _record_llm_call("generate", t_start, time.time(), error=True)
                raise

        llm.generate = traced_generate

        original_query = llm.query

        def traced_query(*args, **kwargs):
            t_start = time.time()
            try:
                result = original_query(*args, **kwargs)
                # llm.query returns (output, req_time, in_tok, out_tok, info)
                in_tok = 0
                out_tok = 0
                if isinstance(result, tuple) and len(result) >= 4:
                    in_tok = result[2] or 0
                    out_tok = result[3] or 0
                _record_llm_call("query", t_start, time.time(), in_tok=in_tok, out_tok=out_tok)
                return result
            except Exception:
                _record_llm_call("query", t_start, time.time(), error=True)
                raise

        llm.query = traced_query
        print("[trace_recorder] llm.generate + llm.query wrapped")
    except Exception as e:
        print(f"[trace_recorder] llm wrap failed: {e}", file=sys.stderr)

    # 2) AgentSearch._run_single_step + execute_deferred_node
    try:
        from engine import agent_search as engine_agent
        original_run_single = engine_agent.AgentSearch._run_single_step

        def traced_run_single(self, parent_node, exec_callback, execute_immediately=True, init_solution_path=None):
            global _pending_step, _step_idx_counter, _current_phase, _current_retry_idx
            step_idx = _step_idx_counter
            _step_idx_counter += 1
            with _lock:
                _pending_step = {
                    "step_idx": step_idx,
                    "step_start_at": _now_rel(),
                    "parent_id": str(getattr(parent_node, "id", "?")),
                    "parent_stage": getattr(parent_node, "stage", None),
                    "agent_used": None,
                    "child_id": None,
                    "branch_id": None,
                    "llm_calls": [],
                    "submissions": [],   # one per try
                }
            _current_retry_idx = 0
            _current_phase = "step"
            try:
                _root, result_node = original_run_single(
                    self, parent_node, exec_callback,
                    execute_immediately=execute_immediately,
                    init_solution_path=init_solution_path,
                )
                with _lock:
                    if result_node is not None:
                        _pending_step["child_id"] = str(getattr(result_node, "id", "?"))
                        _pending_step["agent_used"] = getattr(result_node, "stage", None)
                        _pending_step["branch_id"] = str(getattr(result_node, "branch_id", "?"))
                        _pending_step["is_buggy"] = getattr(result_node, "is_buggy", None)
                        if getattr(result_node, "metric", None) is not None:
                            _pending_step["metric_value"] = getattr(result_node.metric, "value", None)
                    _pending_step["step_end_at"] = _now_rel()
                    _flush_submission(_pending_step)
                    _pending_step = None
                return _root, result_node
            except Exception as exc:
                with _lock:
                    if _pending_step is not None:
                        _pending_step["error"] = repr(exc)
                        _pending_step["step_end_at"] = _now_rel()
                        _flush_submission(_pending_step)
                        _pending_step = None
                raise

        engine_agent.AgentSearch._run_single_step = traced_run_single
        print("[trace_recorder] AgentSearch._run_single_step wrapped")

        if hasattr(engine_agent.AgentSearch, "execute_deferred_node"):
            original_exec_def = engine_agent.AgentSearch.execute_deferred_node

            def traced_exec_def(self, node, exec_callback):
                global _pending_step, _step_idx_counter, _current_retry_idx, _current_phase
                step_idx = _step_idx_counter
                _step_idx_counter += 1
                with _lock:
                    _pending_step = {
                        "step_idx": step_idx,
                        "step_start_at": _now_rel(),
                        "parent_id": str(getattr(node.parent, "id", "?") if node.parent else "root"),
                        "parent_stage": getattr(node.parent, "stage", None) if node.parent else "root",
                        "agent_used": getattr(node, "stage", None),
                        "child_id": str(getattr(node, "id", "?")),
                        "branch_id": str(getattr(node, "branch_id", "?")),
                        "llm_calls": [],
                        "submissions": [],
                        "deferred": True,
                    }
                _current_retry_idx = 0
                _current_phase = "deferred"
                try:
                    result = original_exec_def(self, node, exec_callback)
                    with _lock:
                        if result is not None:
                            _pending_step["is_buggy"] = getattr(result, "is_buggy", None)
                            if getattr(result, "metric", None) is not None:
                                _pending_step["metric_value"] = getattr(result.metric, "value", None)
                        _pending_step["step_end_at"] = _now_rel()
                        _flush_submission(_pending_step)
                        _pending_step = None
                    return result
                except Exception as exc:
                    with _lock:
                        if _pending_step is not None:
                            _pending_step["error"] = repr(exc)
                            _pending_step["step_end_at"] = _now_rel()
                            _flush_submission(_pending_step)
                            _pending_step = None
                    raise

            engine_agent.AgentSearch.execute_deferred_node = traced_exec_def
            print("[trace_recorder] AgentSearch.execute_deferred_node wrapped")
    except Exception as e:
        print(f"[trace_recorder] agent_search wrap failed: {e}", file=sys.stderr)

    # 3) Interpreter.run — capture exec submit/complete + dump code
    try:
        from engine import executor as engine_exec
        original_run = engine_exec.Interpreter.run

        def traced_run(self, code, id, reset_session=True, working_dir=None):
            global _submission_idx_counter
            t_submit = time.time()
            sub_idx = _submission_idx_counter
            _submission_idx_counter += 1
            sub_path = CODE_DIR / f"sub_{sub_idx:03d}.py"
            try:
                sub_path.write_text(code or "")
            except Exception:
                pass
            if _pending_step is not None:
                with _lock:
                    submission_rec = {
                        "submission_idx": sub_idx,
                        "try_idx": _current_retry_idx,
                        "submit_at_real_s": round(t_submit - _t0, 3) if _t0 else 0.0,
                        "code_path": str(sub_path.relative_to(EXP_DIR)),
                        "code_chars": len(code or ""),
                    }
                    _pending_step["submissions"].append(submission_rec)
            try:
                result = original_run(self, code=code, id=id, reset_session=reset_session, working_dir=working_dir)
                t_complete = time.time()
                if _pending_step is not None and _pending_step.get("submissions"):
                    with _lock:
                        last = _pending_step["submissions"][-1]
                        last["exec_complete_at_s"] = round(t_complete - _t0, 3) if _t0 else 0.0
                        last["exec_duration_s_real"] = round(t_complete - t_submit, 3)
                        rc = getattr(result, "rc", getattr(result, "returncode", None))
                        if rc is not None:
                            last["rc"] = int(rc)
                return result
            except Exception:
                t_complete = time.time()
                if _pending_step is not None and _pending_step.get("submissions"):
                    with _lock:
                        last = _pending_step["submissions"][-1]
                        last["exec_complete_at_s"] = round(t_complete - _t0, 3) if _t0 else 0.0
                        last["exec_duration_s_real"] = round(t_complete - t_submit, 3)
                        last["exec_error"] = True
                raise

        engine_exec.Interpreter.run = traced_run
        print("[trace_recorder] Interpreter.run wrapped")
    except Exception as e:
        print(f"[trace_recorder] executor wrap failed: {e}", file=sys.stderr)

    # 4) DebugAgent retry counter (best-effort: tag _current_retry_idx)
    try:
        from agents import debug_agent as debug_mod
        if hasattr(debug_mod, "DebugAgent") and hasattr(debug_mod.DebugAgent, "run"):
            original_debug_run = debug_mod.DebugAgent.run

            def traced_debug_run(self, search, parent_node, *args, **kwargs):
                global _current_retry_idx, _current_phase
                _current_retry_idx = 0
                _current_phase = "debug"
                # Wrap any retry loop by patching the inner _diff_retry method if present
                if hasattr(self, "_diff_retry"):
                    inner = self._diff_retry

                    def traced_inner(*a, **kw):
                        global _current_retry_idx
                        result = inner(*a, **kw)
                        _current_retry_idx += 1
                        return result

                    self._diff_retry = traced_inner
                try:
                    return original_debug_run(self, search, parent_node, *args, **kwargs)
                finally:
                    _current_phase = "step"

            debug_mod.DebugAgent.run = traced_debug_run
            print("[trace_recorder] DebugAgent.run wrapped (retry tagging best-effort)")
    except Exception as e:
        print(f"[trace_recorder] debug_agent wrap failed (non-fatal): {e}", file=sys.stderr)

    print(f"[trace_recorder] all patches installed; trace -> {TRACE_OUT}")
    print(f"[trace_recorder] code dir -> {CODE_DIR}")


if __name__ == "__main__":
    install_patches()
