#!/usr/bin/env bash
# Smoke test: 3-job mini trace × 4 representative configs.
# Verifies scripts/deps work end-to-end before full sweep.
# Total: ~2 min on RTX 5070 Ti.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_DIR="$REPO/experiments/sweep_unbiased_2026-05-06"
PY="$REPO/.venv/bin/python"
[ -x "$PY" ] || PY="python3"

SMOKE_DIR="/tmp/smoke_unbiased"
TRACE="$SMOKE_DIR/smoke_trace.jsonl"
CODE_CACHE="$SMOKE_DIR/codes"
RESULTS="$SMOKE_DIR/results"
mkdir -p "$SMOKE_DIR" "$CODE_CACHE" "$RESULTS"

# Generate 3-job smoke trace via standalone python script (avoids bash heredoc escape pitfalls)
SMOKE_DIR="$SMOKE_DIR" "$PY" "$EXP_DIR/gen_smoke_trace.py"

cleanup_gpu() {
    pkill -9 -f "replay_scheduler.py|replay_torch_mp.py|step_[0-9]\+\.py|nvidia-cuda-mps" 2>/dev/null || true
    echo quit | nvidia-cuda-mps-control 2>/dev/null || true
    sleep 2
}

run_sched_smoke() {
    local id="$1" mode="$2" backend="$3" probe="$4"
    local cfg_dir="$RESULTS/$id"
    mkdir -p "$cfg_dir"
    cleanup_gpu
    rm -rf "/tmp/replay_workdirs/${id}_smoke"
    echo "--- smoke $id  $mode  $backend  $probe ---"
    timeout 180 "$PY" "$EXP_DIR/replay_scheduler.py" \
        --config-id "${id}_smoke" --mode "$mode" --backend "$backend" --batch-search "$probe" \
        --trace "$TRACE" \
        --runtime-root "/tmp/sweep_runtime_${id}_smoke" \
        --results-dir "$cfg_dir/results" \
        --baseline-path "$SMOKE_DIR/baseline.pt" \
        --summary "$cfg_dir/summary.json" \
        --code-cache-dir "$CODE_CACHE" \
        --duration-s 160 > "$cfg_dir/replay.log" 2>&1
    local rc=$?
    local n
    n=$(grep -o '"COMPLETED": *[0-9]*' "$cfg_dir/summary.json" 2>/dev/null | head -1)
    echo "  rc=$rc  $n"
}

run_tmp_smoke() {
    local id="$1" backend="$2" probe="$3"
    local cfg_dir="$RESULTS/$id"
    mkdir -p "$cfg_dir"
    cleanup_gpu
    rm -rf "/tmp/replay_workdirs/${id}_smoke"
    echo "--- smoke $id  torch_mp  $backend  $probe ---"
    timeout 180 "$PY" "$EXP_DIR/replay_torch_mp.py" \
        --config-id "${id}_smoke" --backend "$backend" --batch-search "$probe" \
        --n-workers 2 --trace "$TRACE" \
        --results-dir "$cfg_dir/results" --summary "$cfg_dir/summary.json" \
        --code-cache-dir "$CODE_CACHE" --duration-s 160 \
        > "$cfg_dir/replay.log" 2>&1
    local rc=$?
    local n
    n=$(grep -o '"COMPLETED": *[0-9]*' "$cfg_dir/summary.json" 2>/dev/null | head -1)
    echo "  rc=$rc  $n"
}

# 4 representative configs (one per quadrant of the matrix)
run_sched_smoke B1 serial_basic              exclusive off       # baseline
run_sched_smoke T2 parallel_default          stream    off       # parallel + stream
run_sched_smoke T4 parallel_batch_optimized  mps       binary    # planner + probe (formerly stuck)
run_tmp_smoke   T8 mps binary                                    # torch_mp + mps + probe

echo
echo "=== SMOKE SUMMARY ==="
for id in B1 T2 T4 T8; do
    cfg_dir="$RESULTS/$id"
    n=$(grep -o '"COMPLETED": *[0-9]*' "$cfg_dir/summary.json" 2>/dev/null | head -1 || echo "no_summary")
    echo "  $id: $n"
done
echo "Smoke results in $RESULTS"
