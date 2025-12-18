#!/usr/bin/env bash

set -u

TIMING_LOG="timing.log"

run_and_log () {
    local script="$1"
    local logfile="$2"

    local start end elapsed status

    start=$(date +%s)

    python3 "$script" > "$logfile" 2>&1
    status=$?

    end=$(date +%s)
    elapsed=$(( end - start ))

    {
        echo "$(date '+%Y-%m-%d %H:%M:%S') | $script"
        echo "  Runtime: ${elapsed}s"
        echo "  Exit code: $status"
        if [ "$status" -ne 0 ]; then
            echo "  Status: ERROR"
        else
            echo "  Status: OK"
        fi
        echo
    } >> "$TIMING_LOG"
}

echo "==== Run started at $(date) ====" >> "$TIMING_LOG"

run_and_log run1.py deepwide.log
sleep 5

run_and_log run2.py hybrid.log
sleep 5

run_and_log run3.py deeptree.log

echo "==== Run finished at $(date) ====" >> "$TIMING_LOG"
