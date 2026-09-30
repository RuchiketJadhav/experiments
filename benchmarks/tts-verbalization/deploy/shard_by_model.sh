#!/usr/bin/env bash
# Run every TTS system at once, one process each, then merge.
#
#   bash deploy/shard_by_model.sh [config] [run-root]
#
# The run is ~24 hours serially and almost all of that is waiting on Replicate,
# so the fix is concurrency, not a bigger instance. run.py has no --workers
# yet, but sharding by SYSTEM is safe today and needs no change to the engine:
#
#   * each process writes its own results.jsonl, so there is no concurrent
#     append to one file and no lock to get wrong;
#   * audio stems are {item_id}__{hash(tts_id, revision, text, sample)}, and
#     tts_id is in the hash, so two systems can never write the same filename;
#   * score.py reads results.jsonl and nothing else -- no manifest needed --
#     so a merged directory scores exactly like a single run.
#
# Wall clock becomes the SLOWEST model's time rather than the sum.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

CONFIG="${1:-configs/synthetic_en.yaml}"
ROOT="${2:-runs/syn-en}"

[ -d .venv ] && . .venv/bin/activate
[ -f ./secrets.env ] && { set -a; . ./secrets.env; set +a; }

for var in REPLICATE_API_TOKEN OPENAI_API_KEY; do
  if [ -z "${!var:-}" ]; then
    echo "missing \$$var -- export it before running" >&2
    exit 2
  fi
done

SHARD_DIR="$ROOT/shards"
mkdir -p "$SHARD_DIR"

# One config per enabled system, written from the real config so the model
# version, the revision string and any input overrides are carried verbatim.
IDS="$(python - "$CONFIG" "$SHARD_DIR" <<'PY'
import pathlib, sys, yaml
cfg = yaml.safe_load(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
out = pathlib.Path(sys.argv[2])
ids = []
for entry in cfg.get("tts", []):
    if entry.get("enabled") is False:
        continue
    one = dict(cfg, tts=[entry])
    (out / f"{entry['id']}.yaml").write_text(
        yaml.safe_dump(one, sort_keys=False), encoding="utf-8")
    ids.append(entry["id"])
print(" ".join(ids))
PY
)"
if [ -z "$IDS" ]; then
  echo "no enabled tts systems in $CONFIG" >&2
  exit 2
fi
echo "sharding across: $IDS"

# How many shards run at once. Seven was too many: Replicate rate-limited
# the account and 3,202 recoverable 429s outlasted the retry budget,
# becoming recorded synthesis failures. Three is the safe default; raise it
# only once a run comes back with zero 429s in the shard logs.
MAX_PARALLEL="${MAX_PARALLEL:-3}"

pids=()
for id in $IDS; do
  # Block until a slot frees up.
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do
    sleep 20
  done
  out="$SHARD_DIR/$id"
  mkdir -p "$out"
  nohup python -u run.py --config "$SHARD_DIR/$id.yaml" --out "$out" \
        --retry-errors >> "$out/run.log" 2>&1 &
  pids+=($!)
  echo "  $id -> $out (pid $!)"
  # Stagger the starts. Seven simultaneous cold boots is the one moment the
  # provider is most likely to rate-limit, and a 429 at t=0 costs the whole
  # shard its head start.
  sleep 10
done

echo "waiting for ${#pids[@]} shards..."
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done

echo "== merging =="
mkdir -p "$ROOT/audio" "$ROOT/transcripts"
: > "$ROOT/results.jsonl"
for id in $IDS; do
  out="$SHARD_DIR/$id"
  [ -s "$out/results.jsonl" ] && cat "$out/results.jsonl" >> "$ROOT/results.jsonl"
  # Hard-link rather than copy: same filesystem, and it keeps the merged view
  # from doubling a gigabyte of audio on a 30 GiB volume.
  cp -aln "$out/audio/."       "$ROOT/audio/"       2>/dev/null || true
  cp -aln "$out/transcripts/." "$ROOT/transcripts/" 2>/dev/null || true
done
echo "merged $(wc -l < "$ROOT/results.jsonl") records into $ROOT/results.jsonl"

echo "== scoring =="
python score.py --run "$ROOT" --asr whisper --html "$ROOT/report.html"

if [ "$status" -ne 0 ]; then
  echo
  echo "NOTE: at least one shard exited non-zero. Check $SHARD_DIR/*/run.log."
  echo "Re-running this script resumes: finished items are cached and skipped."
fi
exit "$status"
