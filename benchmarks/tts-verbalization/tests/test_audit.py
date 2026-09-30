"""Listening-audit guards.

An audit exists to check whether the reported numbers are true. That only works
if the audit itself cannot be nudged toward an answer, so the properties worth
testing are the ones that keep it honest:

- the sample is frozen by a seed and cannot be redrawn until it looks good
- a clip appears in exactly one stratum, so no verdict is double counted
- the page leaks neither the system name nor the stratum, because a listener
  who knows a clip is "a flagged-category miss" will hear an artifact

Run: python tests/test_audit.py
"""

import collections
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from audit import STRATA, page_id, render_page, report_verdicts, select, stratify  # noqa: E402

failures: list[str] = []
checks = 0


def check(cond: bool, label: str) -> None:
    global checks
    checks += 1
    if not cond:
        failures.append(label)


SYSTEMS = ["alpha", "bravo", "charlie", "delta"]
# ISBN is the shape asr_suspect flags: the target needs spelled-out letters.
FLAGGED_TARGET = "i s b n nine seven eight three"


def rec(item, sysname, cat, ok, target="one point seven five",
        heard="ONE POINT SEVEN FIVE", wer=0.1):
    return {"key": f"{item}|{sysname}|0|asr", "item_id": item, "tts": sysname,
            "tts_revision": "r", "category": cat, "status": "scored",
            "written": f"written {item}", "target": target, "hypothesis": heard,
            "target_match": ok, "match": ok, "wer": wer,
            "audio": f"{item}__{sysname}.wav", "duration_s": 3.0}


RECS: list[dict] = []
# 6 items every system fails, in a category whose failures are spelled letters
for i in range(6):
    for s in SYSTEMS:
        RECS.append(rec(f"isbn{i}", s, "ISBN", False, FLAGGED_TARGET, "AIASPAN NINE"))
# 6 items every system fails in a clean category -> item or scorer suspect
for i in range(6):
    for s in SYSTEMS:
        RECS.append(rec(f"dec{i}", s, "Decimal", False))
# 6 items where exactly one system fails -> a real difference
for i in range(6):
    for j, s in enumerate(SYSTEMS):
        RECS.append(rec(f"lone{i}", s, "Time", j != 0))
# 4 items everything passes -> control
for i in range(4):
    for s in SYSTEMS:
        RECS.append(rec(f"pass{i}", s, "Cardinal", True))
# 5 hits carrying a bad transcript -> possible false positives
for i in range(5):
    RECS.append(rec(f"fp{i}", "alpha", "Currency", True, wer=0.9))

# ------------------------------------------------------------------ stratify
buckets = stratify(RECS)
check(all(r["category"] == "ISBN" for r in buckets["flagged_miss"]),
      "only recogniser-flagged categories belong in flagged_miss")
check(all(r["category"] == "Decimal" for r in buckets["unanimous_fail"]),
      "unanimous_fail must exclude the flagged categories")
check(len(buckets["lone_fail"]) == 6,
      f"one failing row per lone item, got {len(buckets['lone_fail'])}")
check(all(r["target_match"] is False for r in buckets["lone_fail"]),
      "lone_fail must hold the FAILING row, not its passing siblings")
check(len(buckets["false_positive"]) == 5,
      f"high-WER hits must be collected, got {len(buckets['false_positive'])}")
check(all(r["target_match"] for r in buckets["unanimous_pass"]),
      "the control stratum must contain only passes")

# ------------------------------------------------------------------- select
a = select(RECS, seed=0)
b = select(RECS, seed=0)
c = select(RECS, seed=1)
check([r["key"] for r in a] == [r["key"] for r in b],
      "the same seed must reproduce the same sample")
check([r["key"] for r in a] != [r["key"] for r in c],
      "a different seed must draw a different sample")
check(len({r["key"] for r in a}) == len(a),
      "a clip must not appear twice, or its verdict is double counted")
check(all(r.get("_stratum") in STRATA for r in a),
      "every selected row must carry its stratum for scoring")
check(all(r.get("audio") for r in a), "every selected row must have audio")

# a stratum smaller than its quota must not crash or borrow from another
tiny = select(RECS[:8], seed=0)
check(isinstance(tiny, list), "a thin run must still produce a sample")
check(len({r["key"] for r in tiny}) == len(tiny), "thin runs must not duplicate")

# ------------------------------------------------------------------ blinding
page = render_page(a)
for s in SYSTEMS:
    check(s not in page, f"the page must not name the system ({s})")
for name in STRATA:
    check(name not in page, f"the page must not leak the stratum ({name})")
check("_stratum" not in page, "the stratum key must not reach the DOM")
check(page.count("<audio") == len(a), "every clip needs a player")
check(page.count('data-v="right"') == len(a), "every clip needs a verdict control")

ids = [page_id(r["key"]) for r in a]
check(len(set(ids)) == len(ids), "page ids must not collide")
check(page_id("x|alpha|0|asr") == page_id("x|alpha|0|asr"), "page ids must be stable")
check("alpha" not in page_id("x|alpha|0|asr"), "the page id must not embed the system")

# ------------------------------------------------------------------- scoring
tmp = Path(tempfile.mkdtemp(prefix="ttsaudit-"))
try:
    (tmp / "audit").mkdir()
    rows = [{"key": r["key"], "id": page_id(r["key"]), "stratum": r["_stratum"],
             "tts": r["tts"], "category": r["category"], "target": r.get("target"),
             "hypothesis": r.get("hypothesis"), "written": r["written"]} for r in a]
    (tmp / "audit" / "sample.json").write_text(
        json.dumps({"seed": 0, "exclude": [], "rows": rows}), encoding="utf-8")

    # Every flagged failure was in fact spoken correctly -> claim confirmed.
    verdicts = {r["id"]: ("right" if r["stratum"] == "flagged_miss" else "wrong")
                for r in rows}
    vp = tmp / "audit" / "v.json"
    vp.write_text(json.dumps(verdicts), encoding="utf-8")

    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = report_verdicts(tmp, vp)
    out = buf.getvalue()
    check(code == 0, "scoring a valid verdict file must succeed")
    check("100%" in out, "an all-correct flagged stratum must report 100%")
    check("right to discount" in out,
          "a high correct-share must confirm the artifact claim")
    check("CONTROL FAILED" in out,
          "controls judged 'spoken wrong' must raise the pipeline alarm")

    # The opposite verdict must flip the conclusion, not soften it.
    flipped = {r["id"]: ("wrong" if r["stratum"] == "flagged_miss" else "right")
               for r in rows}
    vp.write_text(json.dumps(flipped), encoding="utf-8")
    buf = io.StringIO()
    with redirect_stdout(buf):
        report_verdicts(tmp, vp)
    out2 = buf.getvalue()
    check("REAL failures" in out2,
          "if flagged clips were spoken wrong the dagger must be called wrong")
    check("SCORING BUGS" in out2,
          "clips spoken correctly but scored wrong must be reported as bugs")

    # Verdicts for clips outside this sample must be ignored, not counted.
    vp.write_text(json.dumps({**flipped, "deadbeef1234": "right"}), encoding="utf-8")
    buf = io.StringIO()
    with redirect_stdout(buf):
        report_verdicts(tmp, vp)
    check(f"{len(rows)} of {len(rows)} clips judged" in buf.getvalue(),
          "unknown keys must not inflate the judged count")

    # An empty verdict file is an error, not a silent zero.
    vp.write_text("{}", encoding="utf-8")
    buf = io.StringIO()
    with redirect_stdout(buf):
        check(report_verdicts(tmp, vp) == 2, "no matching verdicts must fail loudly")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# ------------------------------------------------- end-to-end through build()
# The tests above construct sample.json by hand, which is exactly how a missing
# "id" field survived: every assertion passed while the real build() wrote rows
# that report_verdicts could not match. Drive the actual function instead.
tmp2 = Path(tempfile.mkdtemp(prefix="ttsaudit-e2e-"))
try:
    (tmp2 / "audio").mkdir()
    lines = []
    for r in RECS:
        (tmp2 / "audio" / r["audio"]).write_bytes(b"RIFF----WAVEfmt ")
        lines.append(json.dumps(r))
    (tmp2 / "results.jsonl").write_text("\n".join(lines), encoding="utf-8")

    from audit import build  # noqa: E402
    page_path = build(tmp2, frozenset(), seed=0)
    check(page_path.exists(), "build() must write the page")

    sample = json.loads((tmp2 / "audit" / "sample.json").read_text(encoding="utf-8"))
    check(all("id" in r for r in sample["rows"]),
          "every sample row needs the page id report_verdicts joins on")
    check(all("stratum" in r for r in sample["rows"]),
          "every sample row needs its stratum")

    # The ids in sample.json must be the ids the page actually rendered, and
    # each must have a clip on disk under that same name.
    html = page_path.read_text(encoding="utf-8")
    clips = {p.stem for p in (tmp2 / "audit" / "clips").iterdir()}
    for r in sample["rows"]:
        check(f'data-key="{r["id"]}"' in html, f"page missing card for {r['id']}")
        check(r["id"] in clips, f"no clip on disk for {r['id']}")

    # A verdict file produced against the page must score without hand-editing.
    vp2 = tmp2 / "audit" / "verdicts.json"
    vp2.write_text(json.dumps({r["id"]: "right" for r in sample["rows"]}),
                   encoding="utf-8")
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = report_verdicts(tmp2, vp2)
    check(code == 0, "verdicts straight from the page must score cleanly")
    check(f"{len(sample['rows'])} of {len(sample['rows'])} clips judged"
          in buf.getvalue(),
          "a full verdict file must be recognised in full")

    # Excluding a system must remove it from the sample entirely.
    shutil.rmtree(tmp2 / "audit")
    build(tmp2, frozenset({"alpha"}), seed=0)
    s2 = json.loads((tmp2 / "audit" / "sample.json").read_text(encoding="utf-8"))
    check(all(r["tts"] != "alpha" for r in s2["rows"]),
          "an excluded system must not appear in the audit sample")
    check(s2["exclude"] == ["alpha"], "the sample must record what was excluded")
finally:
    shutil.rmtree(tmp2, ignore_errors=True)

print(f"{checks - len(failures)}/{checks} checks passed")
if failures:
    print(f"\n{len(failures)} FAILED:\n")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("all passed")

# ====================================================== disagreement mode
# Two recognizers scored the same audio and disagreed. The direction of that
# disagreement is the one fact that must never reach the listener: knowing a
# clip is one the second recognizer passed is knowing which answer is
# convenient. Neither transcript may be shown either, since they contradict.
from audit import (DISAGREE_STRATA, build_disagree,  # noqa: E402
                   render_page, report_disagree, select_disagree)

DIS = []
for i in range(30):
    DIS.append({"key": f"d{i}|alpha|0|asr", "item_id": f"d{i}", "tts": "alpha",
                "category": "Time", "written": f"w{i}", "target": "ten thirty",
                "audio": f"d{i}.wav", "direction": "b_passes",
                "whisper_heard": "TEN THIRDY", "deepgram_heard": "ten thirty"})
for i in range(30):
    DIS.append({"key": f"e{i}|bravo|0|asr", "item_id": f"e{i}", "tts": "bravo",
                "category": "ISBN", "written": f"w{i}", "target": "i s b n",
                "audio": f"e{i}.wav", "direction": "a_passes",
                "whisper_heard": "i s b n", "deepgram_heard": "isbeen"})

d1 = select_disagree(DIS, seed=0)
d2 = select_disagree(DIS, seed=0)
d3 = select_disagree(DIS, seed=1)
check([r["key"] for r in d1] == [r["key"] for r in d2],
      "the same seed must reproduce the same disagreement sample")
check([r["key"] for r in d1] != [r["key"] for r in d3],
      "a different seed must draw a different disagreement sample")
check(len({r["key"] for r in d1}) == len(d1), "no clip may be sampled twice")
got = collections.Counter(r["_stratum"] for r in d1)
for name, (want, _why) in DISAGREE_STRATA.items():
    check(got[name] == want, f"{name} should take {want}, took {got[name]}")

# BLINDING: no transcript, no direction, no system
pg = render_page(d1, show_heard=False)
check("grader heard" not in pg,
      "a transcript must not be shown when the recognizers disagree")
for leak in ("TEN THIRDY", "isbeen", "b_passes", "a_passes",
             "direction", "alpha", "bravo", "whisper", "deepgram"):
    check(leak not in pg, f"the page must not leak {leak!r}")
check(pg.count("<audio") == len(d1), "every clip needs a player")
check("Your ear is the only evidence" in pg,
      "the instructions must match a page with no transcript")
# and the normal mode still shows it
check("grader heard" in render_page(a, show_heard=True),
      "single-lineage mode must still show the transcript")

# ------------------------------------------------- build + score round trip
tmp3 = Path(tempfile.mkdtemp(prefix="ttsdis-"))
try:
    (tmp3 / "audio").mkdir()
    for r in DIS:
        (tmp3 / "audio" / r["audio"]).write_bytes(b"RIFF----WAVEfmt ")
    dpath = tmp3 / "disagreement.json"
    dpath.write_text(json.dumps({"a": "whisper", "b": "deepgram",
                                 "rows": DIS}), encoding="utf-8")
    page = build_disagree(tmp3, dpath, seed=0)
    check(page.exists(), "build_disagree must write the page")
    sample = json.loads((tmp3 / "audit-disagree" / "sample.json")
                        .read_text(encoding="utf-8"))
    check(all("id" in r and "stratum" in r for r in sample["rows"]),
          "every sample row needs the id and stratum scoring joins on")
    clips = {p.stem for p in (tmp3 / "audit-disagree" / "clips").iterdir()}
    check(all(r["id"] in clips for r in sample["rows"]),
          "every sampled clip must be on disk under its opaque id")

    # every b_passes clip spoken WRONG => deepgram repaired 100% of them
    v = {r["id"]: ("wrong" if r["stratum"] == "b_passes" else "right")
         for r in sample["rows"]}
    vp3 = tmp3 / "v.json"
    vp3.write_text(json.dumps(v), encoding="utf-8")
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = report_disagree(tmp3, vp3)
    out3 = buf.getvalue()
    check(code == 0, "scoring disagreement verdicts must succeed")
    check("100%" in out3, "an all-wrong b_passes stratum must report 100%")
    check("broken verbalization as correct" in out3,
          "the report must name what a false pass actually means")

    # unclear verdicts are excluded from rates, not counted as either
    v2 = {r["id"]: "unclear" for r in sample["rows"][:5]}
    v2.update({r["id"]: "right" for r in sample["rows"][5:]})
    vp3.write_text(json.dumps(v2), encoding="utf-8")
    buf = io.StringIO()
    with redirect_stdout(buf):
        report_disagree(tmp3, vp3)
    check("5 clip(s) marked unclear" in buf.getvalue(),
          "unclear verdicts must be reported and excluded")
finally:
    shutil.rmtree(tmp3, ignore_errors=True)

print(f"{checks - len(failures)}/{checks} checks passed (disagreement mode)")
if failures:
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
