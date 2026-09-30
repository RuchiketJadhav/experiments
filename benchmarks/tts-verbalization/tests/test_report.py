"""Report rendering guards.

The defect this exists to prevent: render_evidence used to cap each tab at
400 rows. With 10 items that cap never fired, so it shipped. On a full
540-item run across 5 systems it silently hid roughly 550 of ~950 failures
while the section lede claimed to show every scored item. A report that
quietly drops evidence is worse than one that shows none.

Run: python tests/test_report.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from report import (asr_suspect, evidence_omitted,  # noqa: E402
                    render_breakdown, render_categories, render_evidence,
                    render_math)
from score import summarize  # noqa: E402

failures: list[str] = []
checks = 0


def check(cond: bool, label: str) -> None:
    global checks
    checks += 1
    if not cond:
        failures.append(label)


CATS = ["Date", "Time", "Currency", "Address", "Phone", "Ordinal"]


def rec(i: int, sysname: str, ok: bool) -> dict:
    return {"item_id": f"i{i}", "tts": sysname, "tts_revision": "r1",
            "category": CATS[i % len(CATS)], "status": "scored",
            "written": f"written {i}", "target": f"target {i}",
            "hypothesis": f"heard {i}", "target_match": ok, "match": ok,
            "wer": 0.0 if ok else 0.5}


# 5 systems x 250 items = 1250 records, ~40% failures: past the old 400 cap
SYSTEMS = ["alpha", "bravo", "charlie", "delta", "echo"]
RECS = [rec(i, s, (i + j) % 2 != 0)
        for j, s in enumerate(SYSTEMS) for i in range(250)]
SCORED = [r for r in RECS if r["status"] == "scored"]
MISSES = [r for r in SCORED if r["target_match"] is False]

# ------------------------------------------------- evidence is NOT truncated
ev = render_evidence(RECS)
check(ev.count('<tr class="miss"') == len(MISSES),
      f"every failure must render: got {ev.count(chr(60)+'tr class='+chr(34)+'miss'+chr(34))}"
      f" of {len(MISSES)}")
check(ev.count('<tr class="miss"') + ev.count('<tr class="hit"') == len(SCORED),
      "every scored item must render, hits included")
check(len(MISSES) > 400, "fixture must exceed the old 400 cap to be a real guard")

# every row carries the attributes the filter JS reads
import re  # noqa: E402
# data-ok also appears on the three filter buttons, so match the row triple
row_attrs = re.findall(r'data-sys="[^"]*" data-cat="[^"]*" data-ok="[01]"', ev)
check(len(row_attrs) == len(SCORED),
      f"each row needs sys/cat/ok attributes: {len(row_attrs)} of {len(SCORED)}")
check('data-label="Written"' in ev, "mobile stacked-card labels must survive")

# filter controls list every system and category actually present
for s in SYSTEMS:
    check(f"<option>{s}</option>" in ev, f"system filter missing {s}")
for c in CATS:
    check(f"<option>{c}</option>" in ev, f"category filter missing {c}")

# failures render before hits: that is what the reader opens the section for
check(ev.index('<tr class="miss"') < ev.index('<tr class="hit"'),
      "failures must be ordered before correct items")

# ------------------------------------------------------------- breakdown
summary = summarize(RECS)
bd = render_breakdown(summary)
check(bd.count("<article class=\"bd\">") == len(SYSTEMS),
      f"one card per system, got {bd.count(chr(60))}")
check('id="breakdown"' in bd, "breakdown section needs its anchor")
# colour must track the value, not be red at every rate
check("--c:var(--good)" in bd or "--c:#B8862B" in bd,
      "high rates must not render in the failure colour")

# a single-category run has nothing to rank, so the section stays out
one_cat = [r for r in RECS if r["category"] == "Date"]
check(render_breakdown(summarize(one_cat)) == "",
      "breakdown must suppress itself when there is nothing to compare")
check(render_categories(summarize(one_cat)) == "",
      "category matrix must suppress itself with <2 categories")

# ------------------------------------------------- excluded items never score
mixed = RECS[:10] + [{"item_id": "x1", "tts": "alpha", "tts_revision": "r1",
                      "category": "Date", "status": "not_generated",
                      "written": "w", "target": "t", "hypothesis": None,
                      "target_match": None, "match": None, "wer": None}]
ev2 = render_evidence(mixed)
check(ev2.count("<tr class=") == 10,
      "an excluded item must not appear as a scored row")

# ------------------------------------------- the public (--no-evidence) copy
# The ONLY property that makes this file safe to upload: no corpus source text
# survives. The corpus is licensed for use, not redistribution, so a leak here
# is a licensing problem, not a cosmetic one.
omitted = evidence_omitted(len(SCORED))
corpus = {r["written"] for r in RECS} | {r["target"] for r in RECS}
leaked = [c for c in corpus if c and c in omitted]
check(not leaked, f"no corpus text may survive omission; leaked {leaked[:3]}")
check("<tr" not in omitted, "the omitted section must render no item rows")
check("not for redistribution" in omitted,
      "omission must state why, not silently drop the audit trail")
check(f"{len(SCORED):,}" in omitted,
      "the omitted section must still report how many items were scored")

# ------------------------------------------- recogniser-artifact disclosure
# A row where every failure is the recogniser mishearing spelled letters is
# NOT a statement about the systems. Reporting "URL or Email: 0% across all
# five" without saying so is the misleading-number failure this report is
# built to avoid, so the detection is guarded.
def art(cat, target, heard):
    return {"item_id": f"a{id(target)}{heard}", "tts": "alpha", "tts_revision": "r",
            "category": cat, "status": "scored", "written": "w",
            "target": target, "hypothesis": heard, "target_match": False,
            "match": False, "wer": 1.0}


ART = (  # spelled letters the decoder cannot hear
    [art("URL", "h t t p s colon slash slash", "H T T T S COLAND SLASH") for _ in range(4)]
    # homophone: the speech was correct, the decoder had no language model
    + [art("Coord", "forty point seven one two eight", "FORTY POINT SEVEN ONE TO EIGHT")
       for _ in range(4)]
    # a genuine verbalization miss: no artifact present
    + [art("Money", "ten dollars ninety nine cents", "TEN DOLLARS") for _ in range(4)])
sus = asr_suspect(ART)
check(sus["URL"] == 1.0, f"spelled-letter targets must flag: {sus.get('URL')}")
check(sus["Coord"] == 1.0, f"homophone transcripts must flag: {sus.get('Coord')}")
check(sus["Money"] == 0.0, f"a real miss must NOT flag: {sus.get('Money')}")

# a category with no failures must not appear as flagged
clean = [dict(art("Clean", "t", "t"), target_match=True, match=True)]
check("Clean" not in asr_suspect(clean), "categories without failures must not flag")

# the matrix marks flagged rows and prints a visible legend, not just a tooltip
mat = render_categories(summarize(ART), sus)
check(mat.count('<tr class="suspect"') == 2, "flagged rows must be marked")
check("&dagger;" in mat and "class='legend'" in mat,
      "the marker needs a visible legend, a hover tooltip is not enough")
check("not measurable with this recognizer" in mat,
      "the legend must say what the marked rows actually mean")
# an unflagged run gets no legend and no marks
plain = render_categories(summarize(RECS), asr_suspect(RECS))
check("&dagger;" not in plain, "clean runs must not carry artifact marks")

# ------------------------------------------------------ the arithmetic section
# Every figure in this section is computed from the run. The failure it guards
# against is a methodology section that quotes hand-written numbers and quietly
# goes stale the first time the corpus changes.
MATH = [rec(i, "alpha", i % 2 == 0) for i in range(40)]
for i, r in enumerate(MATH):
    r["match"] = r["target_match"] and i % 4 == 0     # strict rate below match rate
msum = summarize(MATH)
math_html = render_math(msum, MATH, asr_suspect(MATH))

k_expected = sum(1 for r in MATH if r["target_match"])
check(f"<span class=\"num\">{k_expected}</span>" in math_html,
      f"numerator must be the real match count ({k_expected})")
check(f"<span class=\"den\">{len(MATH)}</span>" in math_html,
      f"denominator must be items scored ({len(MATH)})")
check(f"{k_expected / len(MATH):.4f}" in math_html,
      "the quoted rate must be derived, not written in")
strict = sum(1 for r in MATH if r["match"])
check(f"{strict}/{len(MATH)}" in math_html,
      "the strict-sentence comparison must use real counts")

# WER formula and bootstrap procedure are both stated
check("S + D + I" in math_html, "the WER formula must be shown")
check("with replacement" in math_html, "the bootstrap must state resampling")
check("2.5th and 97.5th" in math_html, "the percentile rule must be stated")
check("overlap" in math_html, "the reader must be told how to read an interval")

# --------------------------------------------- teaching examples are chosen well
# A worked example built on a function-word target ("the fifteenth of") or on a
# spelled-letter target ("three p m") teaches the wrong lesson, so the picker
# must rank those behind a clean span when one exists.
def mk(sysname, target, heard, ok, sentence_ok):
    return {"item_id": f"e{target}{sysname}", "tts": sysname, "tts_revision": "r",
            "category": "Decimal", "status": "scored", "written": "w",
            "target": target, "hypothesis": heard, "target_match": ok,
            "match": sentence_ok, "wer": 0.1}


PICK = [
    mk("alpha", "the fifteenth of", "THE FIFTEENTH OF AUG", True, False),
    mk("alpha", "three p m", "AT THREE P M", True, False),
    mk("alpha", "one point seven five", "ONE POINT SEVEN FIVE LEADERS", True, False),
]
picked = render_math(summarize(PICK), PICK, {})
check("one point seven five" in picked,
      "a clean content-word span must win the hit slot")
check(">the fifteenth of<" not in picked,
      "a function-word-edged target must not be the worked example")
check(">three p m<" not in picked,
      "a spelled-letter target must not be the worked example")

# degenerate input must not crash the renderer
solo = [mk("alpha", "one point five", "ONE POINT FIVE", True, True)]
check(isinstance(render_math(summarize(solo), solo, {}), str),
      "a run with no failures must still render the arithmetic")

print(f"{checks - len(failures)}/{checks} checks passed")
if failures:
    print(f"\n{len(failures)} FAILED:\n")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("all passed")
