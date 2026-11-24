#!/usr/bin/env bash
# Strata on QFN (Qwen3.8-Flash-Next) with STRATA_DECODE_TIMING / STRATA_VERIFY_PROFILE: where a decode window's time
# goes, isolated and under a CPU interferer (Blink section 2.2 / 6.3 analogue on this machine).
#   ./run_strata_timing.sh [chat|coder|iq2xs|iq2xs-pinned ...]      default: chat coder
#   *-pinned configs: the interferer runs under taskset on PINNED_STRESS_CPUS (default 4-7,12-15) with 8 threads
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
BVS=/home/swamai/Work/experiments/basalt_vs_strata
PY=$BVS/.venv/bin/python
STRESS_THREADS=${STRESS_THREADS:-16}
REPS=${REPS:-2}
CONDS=${CONDS:-"isolated stress"}   # e.g. CONDS=isolated
KEEP_TEXT=${KEEP_TEXT:-}           # KEEP_TEXT=1: save generated text (exactness checks)
models=("$@"); [ ${#models[@]} -eq 0 ] && models=(chat coder)
mkdir -p "$HERE/results/strata"

for m in "${models[@]}"; do
    cfg=$HERE/configs/strata-exact-$m-timing.json
    log=$HERE/logs/engine-strata-exact-$m-timing.log
    : > "$log"
    "$BVS/tools/serve.sh" "$cfg" || { echo "serve failed for $m"; exit 1; }
    for cond in $CONDS; do
        sp=""
        if [ "$cond" = stress ]; then
            case "$m" in
                *-pinned)   # paper Table 3 analogue: the interferer on the cores the engine does not use
                    taskset -c "${PINNED_STRESS_CPUS:-4-7,12-15}" "$HERE/probe/stress" --threads "${PINNED_STRESS_THREADS:-8}" & sp=$! ;;
                *)
                    "$HERE/probe/stress" --threads "$STRESS_THREADS" & sp=$! ;;
            esac
            sleep 2
        fi
        echo "=== $m / $cond" | tee -a "$log"
        "$PY" "$BVS/bench/speed_bench.py" --engine strata-0.2.15 --variant "$m-$cond" --suite decode --reps "$REPS" \
            --out "$HERE/results/strata/$m-$cond.jsonl" --notes "blink: $cond" ${KEEP_TEXT:+--keep-text}
        [ -n "$sp" ] && { kill "$sp"; wait "$sp" 2>/dev/null; }
    done
    "$BVS/tools/serve.sh" stop
    grep -E "^=== |strata decode timing|strata decode GPU stages|generated in" "$log" > "$HERE/results/strata/$m-timing.txt"
done
echo DONE
