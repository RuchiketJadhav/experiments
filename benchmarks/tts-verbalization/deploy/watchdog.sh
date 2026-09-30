#!/usr/bin/env bash
# Restart the run if it stops writing records.
#
# A single Replicate call blocked for 2.2 hours despite per-request timeouts --
# requests' timeout is per socket operation, so a server trickling bytes can
# hold a connection open indefinitely. Rather than guess at the right timeout,
# watch the only thing that matters: whether records are still appearing.
# Resume is free (audio and transcripts are cached), so a needless restart
# costs one item.
#
# Usage:
#   bash watchdog.sh <run-dir> <config> <target-records> [log-file]
#
# TARGET used to be a hardcoded 7562 for one particular run. On an unattended
# cloud run that is a trap: get it wrong and the watchdog either exits before
# the run finishes or never exits at all. deploy/run_bench.sh computes it from
# the config.
cd "$(dirname "$0")" || exit 1

RUN_DIR="${1:-runs/open8}"
CONFIG="${2:-configs/replicate_open10.yaml}"
TARGET="${3:-0}"
LOG="${4:-$RUN_DIR/run.log}"
STALL="${STALL:-420}"          # ~20x the healthy gap between records
RESULTS="$RUN_DIR/results.jsonl"

if [ "$TARGET" -le 0 ]; then
  echo "watchdog: refusing to run without a positive target record count" >&2
  echo "usage: bash watchdog.sh <run-dir> <config> <target-records> [log]" >&2
  exit 2
fi

# Kill the runner without taking out unrelated interpreters. The old line was
# `taskkill //F //IM python.exe`, which on a shared box kills every Python
# process on the machine and does not exist on Linux at all.
kill_runner() {
  if command -v pkill >/dev/null 2>&1; then
    pkill -f "run.py --config $CONFIG" && return 0
  fi
  if command -v taskkill >/dev/null 2>&1; then
    taskkill //F //IM python.exe >/dev/null 2>&1 && return 0
  fi
  return 1
}

echo "$(date +%H:%M:%S) watchdog armed: $RESULTS -> $TARGET records, stall ${STALL}s"

# Condition on the RECORD COUNT, never on manifest.json: that file
# survives from a previous completed run, so the old test was true
# immediately and the watchdog exited without ever guarding anything.
while [ "$(wc -l < "$RESULTS" 2>/dev/null || echo 0)" -lt "$TARGET" ]; do
  sleep 120
  now=$(date +%s)
  mtime=$(stat -c %Y "$RESULTS" 2>/dev/null || echo "$now")
  age=$(( now - mtime ))
  if [ "$age" -gt "$STALL" ]; then
    n=$(wc -l < "$RESULTS" 2>/dev/null || echo 0)
    echo "$(date +%H:%M:%S) STALLED ${age}s at $n records -- restarting"
    kill_runner
    sleep 5
    # Keys come from the environment. On the cloud instance they are injected
    # at launch and secrets.env does not exist; locally it does. Either is fine,
    # neither is required.
    [ -f ./secrets.env ] && { set -a; . ./secrets.env; set +a; }
    nohup python -u run.py --config "$CONFIG" \
      --out "$RUN_DIR" --retry-errors >> "$LOG" 2>&1 &
    sleep 60
  fi
done
echo "$(date +%H:%M:%S) COMPLETE $(wc -l < "$RESULTS") records"
