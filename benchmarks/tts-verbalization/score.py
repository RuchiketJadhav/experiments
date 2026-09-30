"""Aggregate results.jsonl into a readable table and summary.json.

Three rules this file exists to enforce, because each one is a way to publish a
number that is quietly wrong:

1. DENOMINATOR. A rate is scored/generated, never scored/540. Chatterbox
   generated 514 of 540; dividing by 540 would report it as worse than it is
   for reasons that have nothing to do with the model.
2. EXCLUSIONS ARE VISIBLE. not_generated / screened_audio /
   screened_transcript / synthesis_error each get their own column instead of
   being averaged into the score.
3. COMPARABILITY. The cross-system headline is computed on the INTERSECTION of
   items every system generated. Comparing one system on 539 items against
   another on 514 is not like-for-like, so both numbers are shown.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

EXCLUDED = {"not_generated", "screened_audio", "screened_transcript",
            "synthesis_error", "asr_error", "no_target"}


def load_records(path: Path) -> list[dict]:
    recs = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue          # truncated tail from a killed run

    # DEDUPE, keeping the LAST record per key. --retry-errors un-marks a failed
    # key so the item is attempted again, which leaves both the original
    # failure and the retry in the file. Counting both inflates the attempted
    # denominator and can count one item as two outcomes. The later record is
    # the one that reflects what finally happened.
    by_key: dict = {}
    for r in recs:
        k = r.get("key")
        if k is None:
            continue
        by_key[k] = r
    if len(by_key) != len(recs):
        import sys
        print(f"note: collapsed {len(recs) - len(by_key)} superseded retry "
              f"record(s); scoring {len(by_key)} unique items",
              file=sys.stderr)
    return list(by_key.values())


def bootstrap_ci(flags: list[bool], resamples: int = 1000,
                 seed: int = 12345) -> tuple[float, float]:
    """Percentile bootstrap 95% CI on a match rate.

    At 20 items per category the CI is what stops a two-item gap being read as
    a finding.
    """
    if not flags:
        return (0.0, 0.0)
    # Sort first. The resampler draws by INDEX from a fixed seed, so without
    # this the interval depends on the ROW ORDER of the input file: sorting
    # results.jsonl by key shifted every published CI by a few thousandths.
    # A bootstrap interval is a property of the sample, not of the file, and
    # a published number has to survive someone reordering the data.
    # Sorting booleans is a no-op on the multiset the bootstrap draws from.
    flags = sorted(flags)
    rng = random.Random(seed)
    n = len(flags)
    means = []
    for _ in range(resamples):
        means.append(sum(flags[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int(0.025 * resamples)]
    hi = means[min(int(0.975 * resamples), resamples - 1)]
    return (lo, hi)


def summarize(recs: list[dict]) -> dict:
    systems = sorted({r["tts"] for r in recs})
    by_sys: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        by_sys[r["tts"]].append(r)

    # Items every system actually scored -> the like-for-like comparison set.
    scored_ids = {s: {r["item_id"] for r in rs if r.get("status") == "scored"}
                  for s, rs in by_sys.items()}
    common = set.intersection(*scored_ids.values()) if scored_ids else set()

    out: dict = {"systems": {}, "common_items": len(common),
                 "categories": {}, "excluded_reasons": {}}

    for s in systems:
        rs = by_sys[s]
        scored = [r for r in rs if r.get("status") == "scored"]
        flags = [bool(r.get("target_match")) for r in scored]
        rate = (sum(flags) / len(flags)) if flags else 0.0
        lo, hi = bootstrap_ci(flags)
        wers = [r["wer"] for r in scored if r.get("wer") is not None]
        excl = defaultdict(int)
        for r in rs:
            if r.get("status") in EXCLUDED:
                excl[r["status"]] += 1
        common_flags = [bool(r.get("target_match")) for r in scored
                        if r["item_id"] in common]
        out["systems"][s] = {
            "revision": rs[0].get("tts_revision"),
            "scored": len(scored),
            "attempted": len(rs),
            "match_rate": round(rate, 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "mean_wer": round(sum(wers) / len(wers), 4) if wers else None,
            "sentence_match_rate": (round(sum(bool(r.get("match")) for r in scored)
                                          / len(scored), 4) if scored else None),
            "excluded": dict(excl),
            "common_match_rate": (round(sum(common_flags) / len(common_flags), 4)
                                  if common_flags else None),
        }
        for r in scored:
            out["categories"].setdefault(r["category"], {})[s] = None

    # per-category rates
    cat_flags: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for r in recs:
        if r.get("status") == "scored":
            cat_flags[(r["category"], r["tts"])].append(bool(r.get("target_match")))
    for (cat, s), flags in cat_flags.items():
        lo, hi = bootstrap_ci(flags)
        out["categories"].setdefault(cat, {})[s] = {
            "n": len(flags),
            "match_rate": round(sum(flags) / len(flags), 4),
            "ci95": [round(lo, 4), round(hi, 4)],
        }

    reasons = defaultdict(int)
    for r in recs:
        if r.get("status") in EXCLUDED:
            reasons[f"{r['status']}"] += 1
    out["excluded_reasons"] = dict(reasons)
    return out


def print_table(summary: dict) -> None:
    print("\nOVERALL  (match = required verbalization present; sent = strict whole-sentence)")
    print(f"  {'system':<26} {'scored':>7} {'match':>7} {'95% CI':>16} "
          f"{'wer':>6} {'sent':>6} {'common':>8} {'excluded':>9}")
    print(f"  {'-'*26} {'-'*7} {'-'*7} {'-'*16} {'-'*6} {'-'*6} {'-'*8} {'-'*9}")
    for s, d in sorted(summary["systems"].items(),
                       key=lambda kv: -kv[1]["match_rate"]):
        ci = f"[{d['ci95'][0]:.2f}, {d['ci95'][1]:.2f}]"
        wer = f"{d['mean_wer']:.3f}" if d["mean_wer"] is not None else "  -  "
        com = f"{d['common_match_rate']:.2f}" if d["common_match_rate"] is not None else "  -  "
        nex = sum(d["excluded"].values())
        sent = (f"{d['sentence_match_rate']:.2f}"
                if d.get("sentence_match_rate") is not None else "  -  ")
        print(f"  {s:<26} {d['scored']:>7} {d['match_rate']:>7.2f} {ci:>16} "
              f"{wer:>6} {sent:>6} {com:>8} {nex:>9}")
    print(f"\n  like-for-like set: {summary['common_items']} items scored by every system")
    if summary["excluded_reasons"]:
        print(f"  excluded: {summary['excluded_reasons']}")

    cats = summary["categories"]
    if not cats:
        return
    systems = sorted(summary["systems"])
    print("\nBY CATEGORY  (match rate; n per cell)")
    width = max(len(c) for c in cats) + 2
    print("  " + "category".ljust(width) + "".join(f"{s[:14]:>16}" for s in systems))
    for cat in sorted(cats):
        row = "  " + cat.ljust(width)
        for s in systems:
            cell = cats[cat].get(s)
            row += f"{cell['match_rate']:>13.2f}/{cell['n']:<2}" if cell else f"{'-':>16}"
        print(row)


def main() -> int:
    ap = argparse.ArgumentParser(description="Aggregate a tts-bench run")
    ap.add_argument("--run", default="runs/smoke")
    ap.add_argument("--html", nargs="?", const="", default=None,
                    metavar="PATH",
                    help="also write a standalone HTML report "
                         "(default: <run>/report.html)")
    ap.add_argument("--asr", default="", metavar="SUBSTR",
                    help="score only records from the recognizer whose id "
                         "contains this (e.g. whisper). Required once a run "
                         "carries two lineages: averaging them is meaningless")
    ap.add_argument("--exclude", default="", metavar="IDS",
                    help="comma-separated tts ids to leave out of scoring "
                         "and the report (records stay in results.jsonl)")
    ap.add_argument("--no-evidence", action="store_true",
                    help="omit the per-item table from the HTML: the corpus is "
                         "licensed for use but not redistribution, so a copy "
                         "that leaves this machine should not carry it")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    run_dir = Path(args.run)
    results = run_dir / "results.jsonl"
    if not results.exists():
        print(f"no results at {results}", file=sys.stderr)
        return 2
    recs = load_records(results)
    asrs = {r.get("asr") for r in recs}
    if args.asr:
        recs = [r for r in recs if args.asr in (r.get("asr") or "")]
        if not recs:
            print(f"no records from an asr matching {args.asr!r}; "
                  f"run has {sorted(a for a in asrs if a)}", file=sys.stderr)
            return 2
        print(f"scoring recognizer: {sorted({r['asr'] for r in recs})[0]}")
    elif len(asrs) > 1:
        # Two lineages in one file are two measurements of the same audio, not
        # more samples of one. Averaging them hides exactly the cross-family
        # disagreement the second lineage was added to expose.
        names = ", ".join(sorted(a for a in asrs if a))
        print(f"error: this run carries {len(asrs)} recognizers ({names}); "
              f"pass --asr to pick one. Averaging two lineages is not a number.",
              file=sys.stderr)
        return 2
    exclude = frozenset(x.strip() for x in args.exclude.split(",") if x.strip())
    if exclude:
        before = {r["tts"] for r in recs}
        recs = [r for r in recs if r["tts"] not in exclude]
        missed = exclude - before
        if missed:
            print(f"warning: --exclude named unknown system(s): "
                  f"{', '.join(sorted(missed))}", file=sys.stderr)
        print(f"excluded {', '.join(sorted(exclude & before))} "
              f"({len(before)} -> {len({r['tts'] for r in recs})} systems)")
    if not recs:
        print(f"{results} is empty", file=sys.stderr)
        return 2
    summary = summarize(recs)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print_table(summary)
    print(f"\nwrote {run_dir / 'summary.json'}")

    if args.html is not None:
        from report import render
        out = Path(args.html) if args.html else run_dir / "report.html"
        out.write_text(render(run_dir, evidence=not args.no_evidence,
                              exclude=exclude, asr=args.asr),
                       encoding="utf-8")
        print(f"wrote {out}  ({out.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
