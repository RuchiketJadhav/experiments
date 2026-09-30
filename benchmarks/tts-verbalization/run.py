"""Resumable run driver.

Two properties matter more than anything else here for a single operator:

1. CONTENT-HASH AUDIO CACHE. Synthesis is the only step that costs money.
   Cache keys cover the provider, its revision, the text and the sample index,
   so re-scoring after a canonicalizer change re-synthesizes nothing, while a
   config change can never silently reuse audio made under old settings.

2. PER-ITEM ERROR ISOLATION. One guard around each item records the failure
   and continues. Without it a rate-limit at item 500 of 540 loses the run --
   which is the realistic failure, not throughput.

Results are appended to results.jsonl as they are produced, so the file is the
resume state: a restart skips every key already present.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adapters.base import NotGenerated  # noqa: E402
from audio import content_hash, duration_s, file_sha256  # noqa: E402
from canonicalize import (best_wer, has_target, matches,  # noqa: E402
                          target_match, target_text)
from schema import Item  # noqa: E402
from screen import screen_audio, screen_transcript  # noqa: E402


@dataclass
class RunPaths:
    root: Path

    @property
    def audio(self) -> Path: return self.root / "audio"
    @property
    def transcripts(self) -> Path: return self.root / "transcripts"
    @property
    def results(self) -> Path: return self.root / "results.jsonl"
    @property
    def manifest(self) -> Path: return self.root / "manifest.json"

    def ensure(self) -> None:
        for p in (self.root, self.audio, self.transcripts):
            p.mkdir(parents=True, exist_ok=True)


def record_key(item_id: str, tts_id: str, sample: int, asr_id: str) -> str:
    return f"{item_id}|{tts_id}|{sample}|{asr_id}"


# Outcomes caused by the network or a provider, not by the audio. On a long
# unattended run these WILL happen, and treating them as final means a blip at
# hour three becomes a permanent hole that a resume silently skips over.
TRANSIENT = {"asr_error", "synthesis_error"}


def load_done(results_path: Path, retry_errors: bool = False) -> set[str]:
    """Existing result keys. A truncated final line (killed mid-write) is
    dropped rather than crashing the resume.

    With retry_errors, keys whose recorded outcome was TRANSIENT are left out
    of the done set so a resume attempts them again. Deterministic outcomes
    (not_generated, screened_audio, no_target) stay done: retrying those would
    just reproduce the same verdict at the same cost.
    """
    done: set[str] = set()
    if not results_path.exists():
        return done
    with results_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if retry_errors and rec.get("status") in TRANSIENT:
                    continue
                done.add(rec["key"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def run(items: list[Item], tts_adapters: list, asr_adapters: list,
        paths: RunPaths, samples: int = 1, progress: bool = True,
        on_progress=None, retry_errors: bool = False) -> dict:
    """`on_progress(done, total)` is called after each record so a UI can draw
    a bar. Optional; the CLI passes nothing and prints to stderr instead."""
    # build_tts/build_asr already return LISTS. Wrapping one again yields
    # [[adapter]] and fails deep inside the loop with a bare
    # "'list' object has no attribute 'id'". Fail here instead, with the fix.
    for name, adapters in (("tts_adapters", tts_adapters),
                           ("asr_adapters", asr_adapters)):
        for a in adapters:
            if not hasattr(a, "id"):
                raise TypeError(
                    f"{name} must be a flat list of adapters, but contains a "
                    f"{type(a).__name__}. build_tts()/build_asr() already return "
                    f"lists -- pass them straight through rather than wrapping "
                    f"in [ ].")

    paths.ensure()
    done = load_done(paths.results, retry_errors)
    total_expected = (len(items) * max(len(tts_adapters), 1) * max(samples, 1)
                      * max(len(asr_adapters), 1))
    counts = {"written": 0, "skipped_done": 0, "synth_failed": 0,
              "not_generated": 0, "screened_audio": 0, "no_target": 0,
              "screened_transcript": 0, "asr_failed": 0, "cache_hits": 0}
    t0 = time.time()

    with paths.results.open("a", encoding="utf-8") as out:
        for item in items:
            for tts in tts_adapters:
                for sample in range(samples):
                    keys = [record_key(item.id, tts.id, sample, a.id)
                            for a in asr_adapters]
                    if all(k in done for k in keys):
                        counts["skipped_done"] += len(keys)
                        continue

                    stem = f"{item.id}__{content_hash(tts.id, tts.revision, item.written, str(sample))}"
                    wav = paths.audio / f"{stem}.wav"

                    if wav.exists():
                        counts["cache_hits"] += 1
                    else:
                        try:
                            tts.synthesize(item.written, wav,
                                           sample_index=sample, item_id=item.id)
                        except NotGenerated as exc:
                            # Upstream never produced audio (rate limit, SSL
                            # drop). Excluded, not scored as a miss.
                            counts["not_generated"] += 1
                            for asr in asr_adapters:
                                rec = _fail_record(item, tts, sample, asr,
                                                   "not_generated", str(exc))
                                out.write(json.dumps(rec) + "\n")
                                counts["written"] += 1
                            out.flush()
                            continue
                        except Exception as exc:
                            counts["synth_failed"] += 1
                            for asr in asr_adapters:
                                rec = _fail_record(item, tts, sample, asr,
                                                   "synthesis_error",
                                                   f"{type(exc).__name__}: {exc}")
                                out.write(json.dumps(rec) + "\n")
                                counts["written"] += 1
                            out.flush()
                            continue

                    audio_ok = screen_audio(wav, item.written)
                    if not audio_ok:
                        counts["screened_audio"] += 1
                        for asr in asr_adapters:
                            rec = _fail_record(item, tts, sample, asr,
                                               "screened_audio", audio_ok.reason)
                            out.write(json.dumps(rec) + "\n")
                            counts["written"] += 1
                        out.flush()
                        continue

                    for asr in asr_adapters:
                        key = record_key(item.id, tts.id, sample, asr.id)
                        if key in done:
                            counts["skipped_done"] += 1
                            continue
                        try:
                            hyp = _transcribe_cached(asr, wav, paths)
                        except Exception as exc:
                            counts["asr_failed"] += 1
                            rec = _fail_record(item, tts, sample, asr, "asr_error",
                                               f"{type(exc).__name__}: {exc}")
                            out.write(json.dumps(rec) + "\n")
                            # Flushed like the other write paths: an unflushed record
                            # can sit in the buffer when the process is killed, and a
                            # resume would then redo an item already logged.
                            out.flush()
                            counts["written"] += 1
                            continue

                        tx_ok = screen_transcript(hyp, item.spoken)
                        rec = {
                            "key": key, "item_id": item.id, "category": item.category,
                            "source": item.source, "tts": tts.id,
                            "tts_revision": tts.revision, "sample": sample,
                            "asr": asr.id, "asr_revision": asr.revision,
                            "asr_normalizes": getattr(asr, "normalizes_output", None),
                            "written": item.written, "spoken": item.spoken,
                            "hypothesis": hyp, "audio": wav.name,
                            "duration_s": round(duration_s(wav), 3),
                        }
                        if not tx_ok:
                            counts["screened_transcript"] += 1
                            rec.update(status="screened_transcript",
                                       reason=tx_ok.reason, match=None,
                                       target_match=None, wer=None)
                        elif not has_target(item.written, item.spoken, item.category):
                            # Written and spoken are the same words: the item
                            # cannot distinguish a right rendering from a wrong
                            # one, so it is excluded rather than guessed at.
                            counts["no_target"] += 1
                            rec.update(status="no_target",
                                       reason="written and spoken are identical",
                                       match=None, target_match=None, wer=None)
                        else:
                            rec.update(
                                status="scored", reason="",
                                target=target_text(item.written, item.spoken),
                                target_match=target_match(hyp, item.written,
                                                          item.spoken, item.category),
                                match=matches(hyp, item.spoken, item.category),
                                wer=round(best_wer(hyp, item.spoken, item.category), 4),
                            )
                        out.write(json.dumps(rec) + "\n")
                        counts["written"] += 1
                    out.flush()

            if on_progress is not None:
                on_progress(counts["written"] + counts["skipped_done"],
                            total_expected)
            if progress and counts["written"] and counts["written"] % 100 == 0:
                print(f"  {counts['written']} records ({time.time()-t0:.0f}s)",
                      file=sys.stderr)

    counts["elapsed_s"] = round(time.time() - t0, 1)
    return counts


def _fail_record(item: Item, tts, sample: int, asr, status: str, reason: str) -> dict:
    return {
        "key": record_key(item.id, tts.id, sample, asr.id),
        "item_id": item.id, "category": item.category, "source": item.source,
        "tts": tts.id, "tts_revision": tts.revision, "sample": sample,
        "asr": asr.id, "asr_revision": asr.revision,
        "written": item.written, "spoken": item.spoken,
        "hypothesis": None, "status": status, "reason": reason,
        "match": None, "wer": None,
    }


def _transcribe_cached(asr, wav: Path, paths: RunPaths) -> str:
    safe = asr.id.replace("/", "_").replace(":", "_")
    cache = paths.transcripts / f"{wav.stem}__{safe}.txt"
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    hyp = asr.transcribe(wav)
    cache.write_text(hyp, encoding="utf-8")
    return hyp



def _timing_summary(adapter) -> dict | None:
    """Per-system latency, from the timings the adapters already collect.

    They were being thrown away at the end of every run. Latency is not a
    side note here: csm-1b was dropped from the set for 9.9 s/item, which is
    a verdict about whether a model can carry a phone call, and it had to be
    measured by hand because nothing recorded it.

    wall is what the caller waits, including queue and cold boot. predict is
    what the provider bills for. The gap between them is the queue.
    """
    out = {}
    for name in ("wall_times", "predict_times"):
        xs = sorted(getattr(adapter, name, None) or [])
        if not xs:
            continue
        key = name.split("_")[0]
        out[key] = {
            "n": len(xs),
            "mean_s": round(sum(xs) / len(xs), 2),
            "p50_s": round(xs[len(xs) // 2], 2),
            "p90_s": round(xs[min(len(xs) - 1, int(len(xs) * 0.9))], 2),
            "max_s": round(xs[-1], 2),
        }
    return out or None

def write_manifest(paths: RunPaths, items: list[Item], tts_adapters: list,
                   asr_adapters: list, samples: int, dataset_path: Path | None,
                   counts: dict) -> None:
    """Everything needed to interpret a number later.

    Deliberately an allowlist, not a config dump: a dump would put API keys
    from adapter headers into a file on disk.
    """
    manifest = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "items": len(items),
        "categories": sorted({i.category for i in items}),
        "sources": sorted({i.source for i in items}),
        "dataset_sha256": file_sha256(dataset_path) if dataset_path and Path(dataset_path).exists() else None,
        "samples_per_item": samples,
        "tts": [{"id": a.id, "revision": a.revision,
                 "timing_s": _timing_summary(a)} for a in tts_adapters],
        "asr": [{"id": a.id, "revision": a.revision,
                 "normalizes_output": getattr(a, "normalizes_output", None)}
                for a in asr_adapters],
        "counts": counts,
        "note": ("samples are repeat draws, not seeds: hosted TTS APIs expose no "
                 "seed parameter, so repeats measure variance and do not make a "
                 "run reproducible"),
    }
    paths.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def preflight(items: list[Item], tts_adapters: list) -> None:
    """What each system can actually contribute, before spending any compute."""
    all_cats = len({i.category for i in items})
    print(f"dataset: {len(items)} items, {all_cats} categories\n")
    print(f"  {'system':<26} {'generated':>10}  {'categories':>10}  note")
    print(f"  {'-'*26} {'-'*10}  {'-'*10}  {'-'*32}")
    for tts in tts_adapters:
        gen = getattr(tts, "generated_ids", None)
        if gen is None:
            print(f"  {tts.id:<26} {'n/a':>10}  {'n/a':>10}  not a manifest adapter")
            continue
        n = len(gen())
        cats = len(tts.categories_covered())
        note = ""
        if cats < all_cats:
            note = f"PARTIAL: only {cats}/{all_cats} categories"
        elif n < len(items):
            note = f"{len(items) - n} item(s) missing"
        print(f"  {tts.id:<26} {n:>10}  {cats:>3}/{all_cats:<6}  {note}")


def main() -> int:
    ap = argparse.ArgumentParser(description="TTS verbalization bench runner")
    ap.add_argument("--config", default="configs/local_audio.yaml")
    ap.add_argument("--out", default="runs/dev", help="run directory")
    ap.add_argument("--limit", type=int, default=0,
                    help="first N items (NOTE: dataset is category-ordered, so "
                         "--limit 10 is 10 Date items; prefer --sample)")
    ap.add_argument("--sample", type=int, default=0,
                    help="N items spread across all categories (deterministic)")
    ap.add_argument("--seed", type=int, default=0, help="sampling seed")
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--category", action="append", default=[])
    ap.add_argument("--include-partial", action="store_true",
                    help="include systems disabled for low coverage")
    ap.add_argument("--retry-errors", action="store_true",
                    help="on resume, re-attempt items whose recorded outcome "
                         "was a network or provider failure. Recommended for "
                         "long unattended runs: without it a transient blip "
                         "becomes a permanent hole in the data")
    ap.add_argument("--preflight", action="store_true",
                    help="report per-system coverage and exit")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    from config import build_adapters, dataset_path, load_config

    try:
        cfg = load_config(args.config)
        items, tts_adapters, asr_adapters = build_adapters(
            cfg, include_partial=args.include_partial)
    except (FileNotFoundError, ValueError) as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    if not tts_adapters:
        print(f"{args.config} enables no TTS systems.", file=sys.stderr)
        return 2

    if args.preflight:
        preflight(items, tts_adapters)
        return 0

    if args.category:
        wanted = {c.lower() for c in args.category}
        items = [i for i in items if i.category.lower() in wanted]
    if args.sample:
        from sampling import sample_items
        items = sample_items(items, args.sample, seed=args.seed)
    elif args.limit:
        items = items[: args.limit]

    paths = RunPaths(Path(args.out))
    counts = run(items, tts_adapters, asr_adapters, paths,
                 samples=args.samples, retry_errors=args.retry_errors)
    # Hash the corpus this run actually read, not whichever loader happened
    # to be imported at the top of main().
    write_manifest(paths, items, tts_adapters, asr_adapters, args.samples,
                   dataset_path(cfg), counts)
    print(json.dumps(counts, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
