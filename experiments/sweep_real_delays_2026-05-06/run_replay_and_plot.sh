#!/usr/bin/env bash
# SINGLE script for the OTHER MACHINE.
# Sweeps 12 scheduler configs replaying real-trace JSONL with recorded LLM delays,
# then runs plot_results.py to generate 5 PNGs + summary.md.
#
# Usage:
#   bash experiments/sweep_real_delays_2026-05-06/run_replay_and_plot.sh
# Optional env vars:
#   TIME_SCALE   (default 1.0)  scale recorded LLM delays
#   CONFIG_TIMEOUT (default 5400 s = 90 min)  per-config hard cap
#   SKIP_CONFIGS "T4 T5 T6 T7"  space-separated IDs to skip
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_DIR="$REPO/experiments/sweep_real_delays_2026-05-06"
PY="$REPO/.venv/bin/python"
[ -x "$PY" ] || PY="python3"

TRACE="$EXP_DIR/workload_trace_real.jsonl"
CODE_CACHE="$EXP_DIR/replay_codes"
RESULTS_BASE="$REPO/results/sweep_real_delays_2026-05-06"
TIME_SCALE="${TIME_SCALE:-1.0}"
CONFIG_TIMEOUT="${CONFIG_TIMEOUT:-5400}"
SKIP_CONFIGS="${SKIP_CONFIGS:-}"
mkdir -p "$RESULTS_BASE"

if [ ! -f "$TRACE" ]; then
    echo "ERROR: trace file missing: $TRACE"
    echo "Bundled trace was lost; re-run capture_trace.sh on a Claude-SDK-authenticated machine"
    exit 1
fi

n_codes=$(ls "$CODE_CACHE" 2>/dev/null | wc -l)
n_lines=$(wc -l < "$TRACE")
echo "[setup] trace=$TRACE lines=$n_lines codes=$n_codes time_scale=$TIME_SCALE"

cleanup_gpu() {
    pkill -9 -f "replay_scheduler\.py|replay_torch_mp\.py|sub_[0-9]\+\.py|nvidia-cuda-mps" 2>/dev/null || true
    echo quit | nvidia-cuda-mps-control 2>/dev/null || true
    sleep 3
}

should_skip() {
    local id="$1"
    for s in $SKIP_CONFIGS; do
        if [ "$s" = "$id" ]; then return 0; fi
    done
    return 1
}

run_sched() {
    local id="$1" mode="$2" backend="$3" probe="$4"
    if should_skip "$id"; then echo "[$(date -Iseconds)] SKIP $id"; return; fi
    local cfg_dir="$RESULTS_BASE/$id"
    mkdir -p "$cfg_dir"
    cleanup_gpu
    rm -rf "/tmp/replay_workdirs/$id"

    echo "[$(date -Iseconds)] === $id mode=$mode backend=$backend probe=$probe ==="
    nohup nvidia-smi dmon -s pucvmet -d 1 -o T > "$cfg_dir/dmon.csv" 2>&1 &
    local DMON_PID=$!
    sleep 2

    local T0
    T0=$(date +%s.%N)
    timeout "$CONFIG_TIMEOUT" "$PY" "$EXP_DIR/replay_scheduler.py" \
        --config-id "$id" \
        --mode "$mode" --backend "$backend" --batch-search "$probe" \
        --trace "$TRACE" --code-cache-dir "$CODE_CACHE" \
        --runtime-root "/tmp/sweep_runtime_$id" \
        --results-dir "$cfg_dir/results" \
        --baseline-path "$EXP_DIR/baseline.pt" \
        --summary "$cfg_dir/summary.json" \
        --time-scale "$TIME_SCALE" \
        --duration-s $(( CONFIG_TIMEOUT - 60 )) \
        > "$cfg_dir/replay.log" 2>&1
    local RC=$?
    local T1
    T1=$(date +%s.%N)
    local ELAPSED
    ELAPSED=$(echo "$T1 - $T0" | bc -l)
    kill "$DMON_PID" 2>/dev/null
    wait "$DMON_PID" 2>/dev/null
    sleep 1
    echo "$ELAPSED" > "$cfg_dir/wall_clock.txt"
    echo "$RC" > "$cfg_dir/rc.txt"
    echo "[$(date -Iseconds)] $id rc=$RC elapsed=${ELAPSED}s"
}

run_torchmp() {
    local id="$1" backend="$2" probe="$3"
    if should_skip "$id"; then echo "[$(date -Iseconds)] SKIP $id"; return; fi
    local cfg_dir="$RESULTS_BASE/$id"
    mkdir -p "$cfg_dir"
    cleanup_gpu
    rm -rf "/tmp/replay_workdirs/$id"

    echo "[$(date -Iseconds)] === $id torch_mp backend=$backend probe=$probe ==="
    nohup nvidia-smi dmon -s pucvmet -d 1 -o T > "$cfg_dir/dmon.csv" 2>&1 &
    local DMON_PID=$!
    sleep 2

    local T0
    T0=$(date +%s.%N)
    timeout "$CONFIG_TIMEOUT" "$PY" "$EXP_DIR/replay_torch_mp.py" \
        --config-id "$id" --backend "$backend" --batch-search "$probe" \
        --n-workers 2 --trace "$TRACE" --code-cache-dir "$CODE_CACHE" \
        --results-dir "$cfg_dir/results" --summary "$cfg_dir/summary.json" \
        --time-scale "$TIME_SCALE" \
        --duration-s $(( CONFIG_TIMEOUT - 60 )) \
        > "$cfg_dir/replay.log" 2>&1
    local RC=$?
    local T1
    T1=$(date +%s.%N)
    local ELAPSED
    ELAPSED=$(echo "$T1 - $T0" | bc -l)
    kill "$DMON_PID" 2>/dev/null
    wait "$DMON_PID" 2>/dev/null
    sleep 1
    echo "$ELAPSED" > "$cfg_dir/wall_clock.txt"
    echo "$RC" > "$cfg_dir/rc.txt"
    echo "[$(date -Iseconds)] $id rc=$RC elapsed=${ELAPSED}s"
}

# 12 configs
run_sched   B1  serial_basic              exclusive  off
run_sched   B2  serial_batch_optimized    exclusive  binary
run_sched   B3  serial_batch_optimized    exclusive  power_of_two
run_sched   T1  parallel_default          mps        off
run_sched   T2  parallel_default          stream     off
run_sched   T4  parallel_batch_optimized  mps        binary
run_sched   T5  parallel_batch_optimized  mps        power_of_two
run_sched   T6  parallel_batch_optimized  stream     binary
run_sched   T7  parallel_batch_optimized  stream     power_of_two
run_torchmp T8  mps    binary
run_torchmp T9  mps    power_of_two
run_torchmp T10 stream binary
run_torchmp T11 stream power_of_two

echo "[$(date -Iseconds)] === SWEEP COMPLETE — running plots ==="

RESULTS_DIR="$RESULTS_BASE" "$PY" "$EXP_DIR/plot_results.py"

echo
echo "=== final summary ==="
for id in B1 B2 B3 T1 T2 T4 T5 T6 T7 T8 T9 T10 T11; do
    cfg_dir="$RESULTS_BASE/$id"
    rc=$(cat "$cfg_dir/rc.txt" 2>/dev/null || echo "?")
    wc=$(cat "$cfg_dir/wall_clock.txt" 2>/dev/null || echo "?")
    n=$(grep -o '"COMPLETED": *[0-9]*' "$cfg_dir/summary.json" 2>/dev/null | head -1 || echo "no_summary")
    echo "  $id: rc=$rc wall=${wc}s $n"
done
echo
echo "Results in $RESULTS_BASE"
echo "Plots in  $RESULTS_BASE/plots"
