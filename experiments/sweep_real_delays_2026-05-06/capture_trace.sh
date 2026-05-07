#!/usr/bin/env bash
# Phase 1: Capture a real MLEvolve trace driven by Claude opus 4.7 via claude_agent_sdk.
# Runs MLEvolve once on cassava-leaf-disease-classification with the trace_recorder.py
# monkey-patches loaded first to record per-LLM-call delays + per-try executions.
#
# Outputs (relative to this script's directory):
#   workload_trace_real.jsonl     # one JSONL line per MLEvolve step (with submissions[])
#   replay_codes/sub_NNN.py       # one Python script per Interpreter.run() call
#   mlevolve_capture.log          # full MLEvolve stdout/stderr
#
# Usage from repo root:
#   bash experiments/sweep_real_delays_2026-05-06/capture_trace.sh
#
# Estimated runtime: 3-7 hours (20 MLEvolve steps × Claude opus 4.7 latency).
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_DIR="$REPO/experiments/sweep_real_delays_2026-05-06"
PY="$REPO/.venv/bin/python"
[ -x "$PY" ] || PY="python3"

CASSAVA_ROOT="${CASSAVA_ROOT:-/home/downeyflyfan/Research_Projects/AI/Datasets/mle-bench-data/cassava-leaf-disease-classification/prepared/public}"
CASSAVA_PARENT="$(dirname "$(dirname "$(dirname "$CASSAVA_ROOT")")")"   # = mle-bench-data root

cd "$REPO"
export PYTHONPATH="$EXP_DIR:$REPO:${PYTHONPATH:-}"
export TRACE_OUT="$EXP_DIR/workload_trace_real.jsonl"
export CODE_DIR="$EXP_DIR/replay_codes"
export TRACE_RESET=1

mkdir -p "$CODE_DIR"

echo "[$(date -Iseconds)] === MLEvolve real-trace capture starting ==="
echo "  REPO         $REPO"
echo "  CASSAVA_ROOT $CASSAVA_ROOT"
echo "  trace out    $TRACE_OUT"
echo "  code dir     $CODE_DIR"

# Wrapper: install trace patches FIRST, monkeypatch load_cfg to use config_sweep.yaml,
# then hand off to run.py with cassava CLI overrides.
"$PY" -c "
import sys
from pathlib import Path
sys.path.insert(0, '$EXP_DIR')
sys.path.insert(0, '$REPO')
import trace_recorder
trace_recorder.install_patches()

import config as _cfg_mod
from omegaconf import OmegaConf
def _patched_load_cfg():
    base = OmegaConf.load(Path('$REPO/config/config.yaml'))
    override = OmegaConf.load(Path('$REPO/config/config_sweep.yaml'))
    # Drop the 'defaults' key (Hydra-only directive)
    if 'defaults' in override:
        override = {k: v for k, v in override.items() if k != 'defaults'}
        override = OmegaConf.create(override)
    merged = OmegaConf.merge(base, override, OmegaConf.from_cli())
    return _cfg_mod.prep_cfg(merged)
_cfg_mod.load_cfg = _patched_load_cfg

sys.argv = [
    'run.py',
    'exp_id=cassava-leaf-disease-classification',
    'dataset_dir=$CASSAVA_PARENT',
    'data_dir=\${dataset_dir}/cassava-leaf-disease-classification/prepared/public',
    'desc_file=\${data_dir}/description.md',
    'exp_name=cassava_real_trace_2026-05-06',
    'start_cpu_id=0',
    'cpu_number=21',
    'scheduler.enabled=false',   # avoid AF_UNIX path too long; trace_recorder still hooks Interpreter.run
]
import runpy
runpy.run_path('$REPO/run.py', run_name='__main__')
" 2>&1 | tee "$EXP_DIR/mlevolve_capture.log"

RC=$?
echo "[$(date -Iseconds)] === capture finished rc=$RC ==="
echo "trace lines: $(wc -l < "$TRACE_OUT" 2>/dev/null || echo 0)"
echo "code files:  $(ls "$CODE_DIR" 2>/dev/null | wc -l)"
