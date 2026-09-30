#!/usr/bin/env python3
"""Tier 1: prove the published numbers come from the published data.

Free. No API keys. About a minute.

This recomputes every summary from results/records.jsonl.gz using this study's
own score.py, then checks three things:

  1. the recomputed summaries match results/summary-*.json exactly
  2. every number printed in README.md matches the recomputed value
  3. the record ledger (totals, exclusions, disagreements) matches README.md

Check 2 is the one that matters to a stranger. A README is prose and prose
drifts; this makes the prose fail the build when it stops describing the data.

Exit code 0 if everything matches, 1 otherwise.
"""
from __future__ import annotations

import gzip
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from score import bootstrap_ci, summarize  # noqa: E402

RESULTS = HERE / "results"
README = HERE / "README.md"
RECOGNIZERS = {"openai-whisper-1": "summary-whisper-1.json",
               "deepgram-nova-3": "summary-nova-3.json"}

checks: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    checks.append((bool(ok), name, detail))


def load_records() -> list[dict]:
    path = RESULTS / "records.jsonl.gz"
    if not path.exists():
        sys.exit(f"missing {path}. Are you in studies/tts-verbalization?")
    out = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def readme_table(marker: str) -> list[list[str]]:
    """Rows of the markdown table inside <!-- verify:MARKER --> ... <!-- /verify:MARKER -->."""
    text = README.read_text(encoding="utf-8")
    m = re.search(rf"<!-- verify:{marker} -->(.*?)<!-- /verify:{marker} -->",
                  text, re.S)
    if not m:
        sys.exit(f"README.md has no <!-- verify:{marker} --> block")
    rows = []
    for line in m.group(1).splitlines():
        line = line.strip()
        if not line.startswith("|") or set(line) <= set("|-: "):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        rows.append(cells)
    return rows[1:]          # drop the header row


def readme_facts() -> dict[str, str]:
    """Every `key = value` inside the verify:facts block."""
    text = README.read_text(encoding="utf-8")
    m = re.search(r"<!-- verify:facts -->(.*?)<!-- /verify:facts -->", text, re.S)
    if not m:
        sys.exit("README.md has no <!-- verify:facts --> block")
    return dict(re.findall(r"^\s*-\s*`([a-z_0-9]+)`\s*=\s*`([^`]+)`",
                           m.group(1), re.M))


def main() -> int:
    recs = load_records()
    facts = readme_facts()

    # ---------------------------------------------------------- ledger
    check(len(recs) == int(facts.get("unique_records", -1)),
          "record count matches README",
          f"data={len(recs)} readme={facts.get('unique_records')}")

    status = Counter(r["status"] for r in recs)
    for key, label in (("scored", "scored"), ("asr_error", "asr_error"),
                       ("screened_transcript", "screened_transcript")):
        want = facts.get(f"records_{label}")
        if want is not None:
            check(status[key] == int(want), f"{label} count matches README",
                  f"data={status[key]} readme={want}")

    check(len({r['item_id'] for r in recs}) == int(facts.get("items", -1)),
          "item count matches README",
          f"data={len({r['item_id'] for r in recs})} readme={facts.get('items')}")
    check(len({r['category'] for r in recs}) == int(facts.get("categories", -1)),
          "category count matches README",
          f"data={len({r['category'] for r in recs})} readme={facts.get('categories')}")
    check(len({r['tts'] for r in recs}) == int(facts.get("models", -1)),
          "model count matches README",
          f"data={len({r['tts'] for r in recs})} readme={facts.get('models')}")

    # ------------------------------------------ summaries, exact match
    recomputed = {}
    for asr, fname in RECOGNIZERS.items():
        sub = [r for r in recs if r["asr"] == asr]
        got = summarize(sub)
        recomputed[asr] = got
        published = json.loads((RESULTS / fname).read_text(encoding="utf-8"))
        same = all(got["systems"][s] == published["systems"][s]
                   for s in got["systems"]) and \
            set(got["systems"]) == set(published["systems"])
        check(same, f"{fname} reproduces exactly from records.jsonl.gz",
              "" if same else "recomputed summary differs from the published file")

    # ------------------------------- README results table vs recomputed
    for row in readme_table("results"):
        model, wh_rate, wh_ci, dg_rate = row[0], row[1], row[2], row[3]
        w = recomputed["openai-whisper-1"]["systems"].get(model)
        d = recomputed["deepgram-nova-3"]["systems"].get(model)
        if not w or not d:
            check(False, f"README lists a model not in the data: {model}")
            continue
        check(f"{w['match_rate']:.2f}" == wh_rate,
              f"{model}: whisper-1 rate",
              f"data={w['match_rate']:.2f} readme={wh_rate}")
        ci = f"{w['ci95'][0]:.2f}-{w['ci95'][1]:.2f}"
        check(ci == wh_ci, f"{model}: whisper-1 95% CI",
              f"data={ci} readme={wh_ci}")
        check(f"{d['match_rate']:.2f}" == dg_rate,
              f"{model}: nova-3 rate",
              f"data={d['match_rate']:.2f} readme={dg_rate}")

    # ------------- README headline table (the short block at the top)
    for row in readme_table("headline"):
        model, rate, ci = row[0], row[1], row[2]
        w = recomputed["openai-whisper-1"]["systems"].get(model)
        if not w:
            check(False, f"headline table lists an unknown model: {model}")
            continue
        check(f"{w['match_rate']:.2f}" == rate, f"headline: {model} rate",
              f"data={w['match_rate']:.2f} readme={rate}")
        got = f"{w['ci95'][0]:.2f}-{w['ci95'][1]:.2f}"
        check(got == ci, f"headline: {model} 95% CI", f"data={got} readme={ci}")

    # --------------------------------- the hashtag claim in the opening
    hsh = [r for r in recs if r["category"] == "Hashtag or Mention"
           and r["asr"] == "openai-whisper-1" and r["status"] == "scored"]
    check(len(hsh) == int(facts.get("hashtag_scored_whisper", -1)),
          "hashtag clips scored by whisper-1 matches README",
          f"data={len(hsh)} readme={facts.get('hashtag_scored_whisper')}")
    check(sum(1 for r in hsh if r.get("target_match"))
          == int(facts.get("hashtag_correct_whisper", -1)),
          "hashtag correct count matches README",
          f"data={sum(1 for r in hsh if r.get('target_match'))} "
          f"readme={facts.get('hashtag_correct_whisper')}")

    # ------------------------------ README category table vs recomputed
    wh = [r for r in recs
          if r["asr"] == "openai-whisper-1" and r["status"] == "scored"]
    cells = defaultdict(list)
    for r in wh:
        cells[(r["category"], r["tts"])].append(bool(r.get("target_match")))
    best = {}
    for (cat, tts), flags in cells.items():
        rate = sum(flags) / len(flags)
        if cat not in best or rate > best[cat][0]:
            best[cat] = (rate, tts)
    for row in readme_table("categories"):
        cat, claimed = row[0], row[1]
        if cat not in best:
            check(False, f"README names a category not in the data: {cat}")
            continue
        check(f"{best[cat][0]:.2f}" == claimed,
              f"best score for {cat}",
              f"data={best[cat][0]:.2f} readme={claimed}")

    # --------------------------------------------- cross-recognizer set
    dis = RESULTS / "recognizer-disagreements.csv"
    n_dis = sum(1 for _ in dis.open(encoding="utf-8")) - 1
    check(n_dis == int(facts.get("disagreements", -1)),
          "disagreement row count matches README",
          f"data={n_dis} readme={facts.get('disagreements')}")

    by_item = defaultdict(dict)
    for r in recs:
        if r["status"] == "scored":
            by_item[(r["item_id"], r["tts"])][r["asr"]] = bool(r.get("target_match"))
    common = [v for v in by_item.values() if len(v) == 2]
    differ = sum(1 for v in common
                 if v["openai-whisper-1"] != v["deepgram-nova-3"])
    check(len(common) == int(facts.get("common_clips", -1)),
          "clips scored by both recognizers matches README",
          f"data={len(common)} readme={facts.get('common_clips')}")
    check(differ == n_dis,
          "disagreements recomputed from records match the published CSV",
          f"recomputed={differ} csv={n_dis}")

    order_w = sorted(recomputed["openai-whisper-1"]["systems"],
                     key=lambda k: -recomputed["openai-whisper-1"]["systems"][k]["match_rate"])
    order_d = sorted(recomputed["deepgram-nova-3"]["systems"],
                     key=lambda k: -recomputed["deepgram-nova-3"]["systems"][k]["match_rate"])
    check(order_w == order_d,
          "the two recognizers rank the models identically",
          f"whisper={order_w} nova={order_d}")

    # ------------------------------------------------------- category CSV
    cat_csv = RESULTS / "category-results.csv"
    n_cat = sum(1 for _ in cat_csv.open(encoding="utf-8")) - 1
    expect = len({r["category"] for r in recs}) * len({r["tts"] for r in recs}) * 2
    check(n_cat == expect, "category-results.csv is complete",
          f"rows={n_cat} expected={expect}")

    # ------------------------------------------------------------ report
    width = max(len(n) for _, n, _ in checks) + 2
    print()
    print("  Tier 1 verification: recomputing published numbers from published data")
    print("  " + "-" * (width + 46))
    failed = 0
    for ok, name, detail in checks:
        mark = "PASS" if ok else "FAIL"
        line = f"  {mark}  {name:<{width}}"
        if not ok or (detail and "readme=" in detail and not ok):
            line += f"  {detail}"
        print(line)
        if not ok:
            failed += 1
    print("  " + "-" * (width + 46))
    print(f"  {len(checks) - failed} of {len(checks)} checks passed")
    if failed:
        print(f"\n  {failed} FAILED. The published numbers do not match the "
              f"published data, or README.md has drifted from both.")
        return 1
    print("\n  Everything in README.md is reproduced from results/records.jsonl.gz.")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
