#!/usr/bin/env bash
# Run the benchmark unattended on the instance, with the watchdog armed.
#
#   bash deploy/run_bench.sh [config] [run-dir]
#
# Expects the API keys in the environment already (see bootstrap.sh). It will
# fall back to ./secrets.env if that file exists, which is the local
# development path; on the instance there should be no such file.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

CONFIG="${1:-configs/synthetic_en.yaml}"
RUN_DIR="${2:-runs/syn-en}"
LOG="$RUN_DIR/run.log"

[ -d .venv ] && . .venv/bin/activate
[ -f ./secrets.env ] && { set -a; . ./secrets.env; set +a; }

for var in REPLICATE_API_TOKEN OPENAI_API_KEY; do
  if [ -z "${!var:-}" ]; then
    echo "missing \$$var -- export it or fetch it from SSM before running" >&2
    exit 2
  fi
done

mkdir -p "$RUN_DIR"

# The expected record count comes from the config, not from a number typed
# into the watchdog. items x enabled systems x recognizers x samples.
TARGET="$(python - "$CONFIG" <<'PY'
import sys
from config import build_adapters, load_config
cfg = load_config(sys.argv[1])
items, tts, asr = build_adapters(cfg)
samples = int(cfg.get("samples", 1))
print(len(items) * len(tts) * len(asr) * samples)
PY
)"
if ! [ "${TARGET:-0}" -gt 0 ] 2>/dev/null; then
  echo "could not compute the expected record count from $CONFIG" >&2
  exit 2
fi
echo "expecting $TARGET records in $RUN_DIR/results.jsonl"

nohup python -u run.py --config "$CONFIG" --out "$RUN_DIR" \
      >> "$LOG" 2>&1 &
RUNNER=$!
echo "runner pid $RUNNER, log $LOG"

# The watchdog blocks until the target is reached, so this script is the thing
# to run under nohup/systemd and watch.
bash watchdog.sh "$RUN_DIR" "$CONFIG" "$TARGET" "$LOG"

echo "== scoring =="
python score.py --run "$RUN_DIR" --asr whisper --html "$RUN_DIR/report.html"

if [ -n "${S3_BUCKET:-}" ]; then
  echo "== syncing to s3://$S3_BUCKET =="
  # Results and the report always. Audio only when asked: it is ~1 GB for
  # seven systems over 1,080 items and is only needed for a listening audit.
  aws s3 sync "$RUN_DIR" "s3://$S3_BUCKET/$(basename "$RUN_DIR")" \
      ${SYNC_AUDIO:+} ${SYNC_AUDIO:---exclude 'audio/*'}
fi

echo "done."
