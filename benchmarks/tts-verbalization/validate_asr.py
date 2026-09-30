"""Measure a recognizer against human ears before trusting its numbers.

A 40-clip listening audit produced verdicts on what the SPEECH actually said.
Those verdicts are a property of the audio, so they are recognizer-independent
and reusable forever: every candidate recognizer can be scored against the same
ground truth for the price of 40 transcriptions.

TWO NUMBERS DECIDE WHETHER A RECOGNIZER IS USABLE.

1. Agreement: how often the pipeline's verdict matches the human's.
2. FALSE POSITIVES: clips a human judged spoken WRONG that the pipeline scores
   as passing. This is the number that must be zero. A recognizer that applies
   inverse text normalization can rewrite "five twenty twenty twenty three"
   into "5/20/2023", repairing a broken system into a passing one, and no
   amount of agreement elsewhere compensates for a benchmark that flatters
   failures.

Measured so far, on the same 40 clips:

    wav2vec2-base-960h    48%   0 false positives
    whisper-small.en      75%   0 false positives   (local, fp32)

A candidate must reach >= 75% with 0 false positives to be used.

NOTE ON THE CEILING. The sample was stratified from wav2vec2's failures, so it
is a deliberately hard set and these percentages are a comparative score across
recognizers, NOT an estimate of accuracy over the whole corpus.

    set -a; . ./secrets.env; set +a
    python validate_asr.py --verdicts ~/Downloads/verdicts.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from canonicalize import target_match  # noqa: E402
from config import build_asr, load_config  # noqa: E402

FLOOR_AGREEMENT = 0.75


def load_ground_truth(run_dir: Path, verdicts_path: Path):
    sample = json.loads((run_dir / "audit" / "sample.json").read_text(encoding="utf-8"))
    verdicts = json.loads(verdicts_path.read_text(encoding="utf-8"))
    recs = {}
    for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            recs[r["key"]] = r
    # Rows the audit drew may have been rescored since; the AUDIO and the human
    # verdict are what matter, so join on the key and take the record for its
    # written/spoken/target fields.
    out = []
    for row in sample["rows"]:
        v = verdicts.get(row["id"])
        rec = recs.get(row["key"])
        if v in ("right", "wrong") and rec and rec.get("audio"):
            out.append((rec, v))
    return out


def evaluate(adapter, rows, run_dir: Path, cache: Path) -> dict:
    """Transcribe every clip once and compare the pipeline verdict to the human.

    Transcripts are cached per (adapter, clip) so re-running after a
    canonicalize change costs nothing: the expensive half is the API call, the
    interesting half is the scoring, and they should not be coupled.
    """
    cache.mkdir(parents=True, exist_ok=True)
    detail, errors, t0 = [], [], time.time()
    for rec, human in rows:
        cf = cache / f"{adapter.id.replace('/', '_').replace(':', '_')}__{rec['audio']}.txt"
        if cf.exists():
            hyp = cf.read_text(encoding="utf-8")
        else:
            try:
                hyp = adapter.transcribe(run_dir / "audio" / rec["audio"])
            except Exception as exc:                      # noqa: BLE001
                # Per-clip isolation, matching run.py. A DNS blip on clip 34
                # once discarded 33 paid transcriptions and the whole
                # validation; a transient network fault must cost one clip.
                errors.append(f"{rec['audio']}: {type(exc).__name__}")
                continue
            cf.write_text(hyp, encoding="utf-8")
        scored = bool(target_match(hyp, rec["written"], rec["spoken"], rec["category"]))
        detail.append({"key": rec["key"], "human": human, "scored": scored,
                       "category": rec["category"], "target": rec.get("target"),
                       "hypothesis": hyp, "tts": rec["tts"]})
    if errors:
        print(f"    {len(errors)} clip(s) could not be transcribed, excluded "
              f"from the rates: {errors[:3]}")

    right = [d for d in detail if d["human"] == "right"]
    wrong = [d for d in detail if d["human"] == "wrong"]
    tp = sum(1 for d in right if d["scored"])
    tn = sum(1 for d in wrong if not d["scored"])
    fp = [d for d in wrong if d["scored"]]
    return {"id": adapter.id, "n": len(detail), "errors": errors,
            "elapsed_s": time.time() - t0,
            "normalizes": getattr(adapter, "normalizes_output", None),
            "right_n": len(right), "wrong_n": len(wrong),
            "correct_passes": tp, "wrong_fails": tn,
            "false_positives": fp,
            "agreement": (tp + tn) / len(detail) if detail else 0.0,
            "detail": detail}


def report(results: list[dict]) -> bool:
    print()
    print(f"  {'recognizer':<24}{'agree':>8}{'spoken ok':>12}"
          f"{'spoken wrong':>14}{'false pos':>11}{'verdict':>10}")
    print(f"  {'-' * 24}{'-' * 8}{'-' * 12}{'-' * 14}{'-' * 11}{'-' * 10}")
    ok_all = True
    for r in results:
        passes = r["agreement"] >= FLOOR_AGREEMENT and not r["false_positives"]
        ok_all &= passes
        ok_col = f"{r['correct_passes']}/{r['right_n']}"
        bad_col = f"{r['wrong_fails']}/{r['wrong_n']}"
        print(f"  {r['id'][:24]:<24}{r['agreement']:>7.0%} {ok_col:>11}"
              f"{bad_col:>14}{len(r['false_positives']):>11}"
              f"{'PASS' if passes else 'FAIL':>10}")
    print(f"\n  bar: >= {FLOOR_AGREEMENT:.0%} agreement AND 0 false positives")

    for r in results:
        if r["false_positives"]:
            print(f"\n  {r['id']} FALSE POSITIVES -- spoken wrong, "
                  f"scored as passing.")
            print("  This is the disqualifying error: the recognizer is repairing")
            print("  broken speech into a passing score.")
            for d in r["false_positives"][:6]:
                print(f"    {d['category']:<24} want {d['target']!r}")
                print(f"    {'':<24} got  {d['hypothesis']!r}")
    return ok_all


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate recognizers against human verdicts")
    ap.add_argument("--run", default="runs/full")
    ap.add_argument("--config", default="configs/local_audio.yaml")
    ap.add_argument("--verdicts", required=True)
    ap.add_argument("--only", default="",
                    help="comma-separated substrings; default is every asr in "
                         "the config including disabled ones")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    run_dir = Path(args.run)
    rows = load_ground_truth(run_dir, Path(args.verdicts).expanduser())
    if not rows:
        print("no labelled clips found", file=sys.stderr)
        return 2
    print(f"ground truth: {len(rows)} human-judged clips "
          f"({sum(1 for _, v in rows if v == 'right')} spoken correctly, "
          f"{sum(1 for _, v in rows if v == 'wrong')} spoken wrong)")

    cfg = load_config(Path(args.config))
    for e in cfg.get("asr", []):
        # Validation is exactly when a not-yet-trusted recognizer is still
        # disabled in the config, so build them all and filter by --only.
        e.pop("enabled", None)
    adapters = build_asr(cfg)
    want = [w.strip() for w in args.only.split(",") if w.strip()]
    if want:
        adapters = [a for a in adapters if any(w in a.id for w in want)]
    if not adapters:
        print(f"no recognizer matched {args.only!r}", file=sys.stderr)
        return 2

    results = []
    for a in adapters:
        print(f"  transcribing {len(rows)} clips with {a.id} ...", flush=True)
        results.append(evaluate(a, rows, run_dir, run_dir / "audit" / "validate"))
        print(f"    {results[-1]['elapsed_s']:.0f}s")

    ok = report(results)
    out = run_dir / "audit" / "validation.json"
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\n  wrote {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
