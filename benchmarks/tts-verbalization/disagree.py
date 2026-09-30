"""Where two recognizer families disagree, and whether the ranking survives.

arXiv:2607.08256 reports that TTS rankings REVERSE across ASR families. Every
result in this project is scored by one family, so that claim is the single
largest threat to it and the one caveat every writeup carries. This measures it
instead of disclosing it.

WHY THE DISAGREEMENT SET IS THE INTERESTING ONE.

Asked how often Deepgram's language model "repairs" a broken verbalization into
a passing one, the honest answer was that no amount of pattern-matching over
the corpus can say. A regex over known abbreviations gave 7%, which was
arbitrary and far too narrow. Counting items whose target introduces a word
absent from the written form gave 100%, which is true and useless -- it counts
the OPPORTUNITY to repair, not the RATE.

Where the two recognizers agree, nothing is in question. Where they disagree,
exactly one is wrong, and which one is the whole question. That set is small,
self-selecting, and needs no detector: it replaces an estimate with a count,
and hands a human ear the only clips worth listening to.

    python disagree.py --run runs/open8 --a whisper --b deepgram
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from score import bootstrap_ci, load_records  # noqa: E402


def split_lineages(recs: list[dict], a: str, b: str):
    """Index both lineages by (item, tts). Only rows SCORED by both can
    disagree: an item one lineage screened out or never generated is a
    different kind of gap and would masquerade as disagreement."""
    idx: dict[str, dict[tuple[str, str], dict]] = {a: {}, b: {}}
    for r in recs:
        asr = r.get("asr") or ""
        for tag in (a, b):
            if tag in asr:
                idx[tag][(r["item_id"], r["tts"])] = r
    common = set(idx[a]) & set(idx[b])
    scored = {k for k in common
              if idx[a][k].get("status") == "scored"
              and idx[b][k].get("status") == "scored"}
    return idx, scored


def rank(idx_side: dict, keys: set) -> list[tuple[str, float, tuple]]:
    by_sys: dict[str, list[bool]] = defaultdict(list)
    for k in keys:
        by_sys[k[1]].append(bool(idx_side[k].get("target_match")))
    out = []
    for s, flags in by_sys.items():
        rate = sum(flags) / len(flags) if flags else 0.0
        out.append((s, rate, bootstrap_ci(flags)))
    return sorted(out, key=lambda t: -t[1])


def main() -> int:
    ap = argparse.ArgumentParser(description="Cross-family disagreement")
    ap.add_argument("--run", default="runs/open8")
    ap.add_argument("--a", default="whisper", help="substring of the SCORING lineage")
    ap.add_argument("--b", default="deepgram", help="substring of the second opinion")
    ap.add_argument("--out", default="", help="write the disagreement set here")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    run_dir = Path(args.run)
    recs = load_records(run_dir / "results.jsonl")
    idx, keys = split_lineages(recs, args.a, args.b)
    if not keys:
        print(f"no items scored by BOTH {args.a!r} and {args.b!r}; "
              f"run has {sorted({r.get('asr') for r in recs})}", file=sys.stderr)
        return 2

    A, B = idx[args.a], idx[args.b]
    print(f"\n{len(keys):,} items scored by both lineages "
          f"({len({k[1] for k in keys})} systems)\n")

    # ---------------------------------------------------- does the ranking hold
    ra, rb = rank(A, keys), rank(B, keys)
    order_a = [s for s, _, _ in ra]
    order_b = [s for s, _, _ in rb]
    print(f"  {'system':<26}{args.a:>10}{args.b:>11}{'delta':>8}{'rank':>7}")
    print(f"  {'-'*26}{'-'*10}{'-'*11}{'-'*8}{'-'*7}")
    rates_b = dict((s, r) for s, r, _ in rb)
    for pos, (s, rate, _) in enumerate(ra):
        moved = order_b.index(s) - pos
        arrow = "same" if moved == 0 else f"{moved:+d}"
        print(f"  {s:<26}{rate:>10.2f}{rates_b[s]:>11.2f}"
              f"{rates_b[s]-rate:>+8.2f}{arrow:>7}")

    same = order_a == order_b
    print(f"\n  RANKING: {'IDENTICAL across both families' if same else 'CHANGES across families'}")
    if not same:
        print(f"    {args.a}: {' > '.join(order_a)}")
        print(f"    {args.b}: {' > '.join(order_b)}")

    # ------------------------------------------------------- the disagreements
    dis = [k for k in keys
           if bool(A[k].get("target_match")) != bool(B[k].get("target_match"))]
    b_only = [k for k in dis if B[k].get("target_match")]      # b passes, a fails
    a_only = [k for k in dis if A[k].get("target_match")]      # a passes, b fails
    print(f"\n  disagreements: {len(dis):,} of {len(keys):,} "
          f"({len(dis)/len(keys):.0%})")
    print(f"    {args.b} passes where {args.a} fails : {len(b_only):,}"
          f"   <- candidate {args.b} REPAIR, or {args.a} mishearing")
    print(f"    {args.a} passes where {args.b} fails : {len(a_only):,}"
          f"   <- candidate {args.b} mishearing")
    print("\n  Neither side is 'right' here. Only listening settles which.")

    # --------------------------------------------------- where they disagree
    by_cat: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for k in keys:
        cat = A[k]["category"]
        by_cat[cat][1] += 1
        if bool(A[k].get("target_match")) != bool(B[k].get("target_match")):
            by_cat[cat][0] += 1
    print(f"\n  {'category':<32}{'disagree':>10}{'of':>6}{'rate':>8}")
    print(f"  {'-'*32}{'-'*10}{'-'*6}{'-'*8}")
    for cat, (d, n) in sorted(by_cat.items(), key=lambda kv: -kv[1][0] / max(kv[1][1], 1))[:10]:
        print(f"  {cat:<32}{d:>10}{n:>6}{d/n:>8.0%}")

    out = Path(args.out) if args.out else run_dir / "disagreement.json"
    out.write_text(json.dumps({
        "a": args.a, "b": args.b, "n_common": len(keys),
        "ranking_identical": same, "order_a": order_a, "order_b": order_b,
        "rows": [{
            "key": A[k]["key"], "item_id": k[0], "tts": k[1],
            "category": A[k]["category"], "written": A[k]["written"],
            "target": A[k].get("target"), "audio": A[k].get("audio"),
            "direction": "b_passes" if B[k].get("target_match") else "a_passes",
            f"{args.a}_heard": A[k].get("hypothesis"),
            f"{args.b}_heard": B[k].get("hypothesis"),
        } for k in dis],
    }, indent=1), encoding="utf-8")
    print(f"\n  wrote {out}  ({len(dis):,} rows for the listening pass)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
