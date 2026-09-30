"""Listen to the audio and check whether the scores are telling the truth.

The report claims a large share of failures are RECOGNIZER artifacts rather
than verbalization errors. That claim was inferred from the shape of the target
and the transcript. It was never checked against the audio, and the audio is
right there: every scored row carries the wav that produced it.

Three rules this file exists to enforce, because each is a way to run an audit
that confirms whatever you already believed:

1. SAMPLE BY INFORMATION, NOT BY FREQUENCY. Listening to 1,649 failures is not
   an option and listening to a random 40 of them is close to useless: most
   failures are unremarkable and confirm nothing. The strata below are chosen
   so that each verdict can overturn a specific conclusion.
2. BLIND THE LISTENER. The page never shows the system name or which stratum a
   clip came from. Knowing a clip is "a flagged-category miss" primes you to
   hear an artifact, which is the hypothesis under test.
3. THE SAMPLE IS FROZEN BY A SEED. Same seed, same 40 rows. An audit you can
   redraw until you like the answer is not an audit.

    python audit.py --run runs/full --exclude smallest-lightning-v3.1
    python audit.py --run runs/full --score runs/full/audit/verdicts.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from report import asr_suspect, esc  # noqa: E402
from score import load_records  # noqa: E402

# Stratum -> (label for the operator, how many clips, what a verdict decides).
# Sizes are deliberate: stratum 1 carries the claim under test and gets the
# most clips; stratum 5 is a control and needs only enough to catch a pipeline
# that is broken outright.
STRATA = {
    "flagged_miss": (16, "Failures in categories the report blames on the "
                         "recognizer. Said correctly here means the report is "
                         "right to discount them."),
    "unanimous_fail": (8, "Every system failed the same item outside those "
                          "categories. Points at the item or the scorer, not "
                          "the systems."),
    "lone_fail": (8, "One system failed where the others passed. Should be a "
                     "real difference; if it is not, the ranking is noise."),
    "false_positive": (5, "Scored correct despite a poor transcript. These "
                          "inflate a score and are the costliest error here."),
    "unanimous_pass": (3, "Everything passed. Control: if these sound wrong, "
                          "nothing above can be trusted."),
}


def stratify(recs: list[dict]) -> dict[str, list[dict]]:
    """Bucket scored rows by what a listening verdict would settle."""
    suspect = asr_suspect(recs)
    flagged = {c for c, v in suspect.items() if v >= 0.5}
    scored = [r for r in recs if r.get("status") == "scored" and r.get("audio")]

    by_item: dict[str, list[dict]] = defaultdict(list)
    for r in scored:
        by_item[r["item_id"]].append(r)
    # Only items every system scored can support a unanimity claim; a 3-of-4
    # item looks unanimous for reasons that have nothing to do with the audio.
    n_sys = len({r["tts"] for r in scored})

    out: dict[str, list[dict]] = {k: [] for k in STRATA}
    for rows in by_item.values():
        misses = [r for r in rows if r.get("target_match") is False]
        full = len(rows) == n_sys
        if full and len(misses) == len(rows):
            key = ("flagged_miss" if rows[0]["category"] in flagged
                   else "unanimous_fail")
            out[key].extend(rows)
        elif full and not misses:
            out["unanimous_pass"].extend(rows)
        elif full and len(misses) == 1:
            out["lone_fail"].extend(misses)
        else:
            out["flagged_miss"].extend(
                m for m in misses if m["category"] in flagged)
    out["false_positive"] = [r for r in scored
                             if r.get("target_match") and (r.get("wer") or 0) > 0.5]
    return out


def select(recs: list[dict], seed: int = 0) -> list[dict]:
    """A frozen, stratified sample. Each row is tagged with its stratum for
    scoring later; the tag never reaches the page."""
    buckets = stratify(recs)
    rng = random.Random(seed)
    picked: list[dict] = []
    seen: set[str] = set()
    for name, (want, _why) in STRATA.items():
        pool = [r for r in buckets[name] if r["key"] not in seen]
        pool.sort(key=lambda r: r["key"])          # deterministic before sampling
        take = rng.sample(pool, min(want, len(pool)))
        for r in take:
            seen.add(r["key"])
            picked.append({**r, "_stratum": name})
    rng.shuffle(picked)                            # interleave so order leaks nothing
    return picked


def page_id(key: str) -> str:
    """Opaque per-clip id for the page.

    The record key is "<item>|<tts>|<sample>|<asr>", so putting it in the DOM
    would print the system name 40 times in the page source and undo the
    blinding entirely. Hash it; sample.json holds the mapping back.
    """
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


def heard_row(r: dict, show: bool) -> str:
    """The grader's transcript, or nothing.

    In disagreement mode two recognizers produced DIFFERENT verdicts on this
    clip, so printing either transcript tells the listener which way to lean.
    The only honest prompt there is the audio and the required span.
    """
    if not show:
        return ""
    return ('<div><dt>the grader heard</dt>'
            f'<dd class="heard">{esc(r.get("hypothesis") or "")}</dd></div>')


def render_page(rows: list[dict], show_heard: bool = True,
                namespace: str = "default") -> str:
    # The instructions have to match what is actually on screen. With no
    # transcript shown, "ignore what the grader heard" is nonsense.
    lede_extra = (" Ignore what the grader heard, that is what is being tested."
                  if show_heard else
                  " No transcript is shown here on purpose: two recognizers "
                  "disagreed about this clip, so either one would tell you "
                  "which way to lean. Your ear is the only evidence.")
    cards = []
    for i, r in enumerate(rows, 1):
        cards.append(f"""
    <article class="clip" data-key="{page_id(r["key"])}">
      <header><span class="n">{i} of {len(rows)}</span>
        <span class="cat">{esc(r['category'])}</span></header>
      <audio controls preload="none" src="clips/{page_id(r["key"])}.wav"></audio>
      <dl>
        <div><dt>text given to the system</dt><dd>{esc(r['written'])}</dd></div>
        <div><dt>it should say</dt><dd class="want">{esc(r.get('target') or '')}</dd></div>
        {heard_row(r, show_heard)}
      </dl>
      <div class="vote">
        <button data-v="right">Said it correctly</button>
        <button data-v="wrong">Said it wrong</button>
        <button data-v="unclear">Can't tell</button>
      </div>
    </article>""")

    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Listening audit</title>
<style>
:root{{--paper:#FBFBF9;--card:#fff;--ink:#1A1A17;--ink-2:#4A4A42;--muted:#6E6E66;
  --rule:#E4E3DC;--accent:#16556B;--good:#1F7A4D;--bad:#A8321F;
  --f-body:"IBM Plex Sans",system-ui,sans-serif;--f-mono:ui-monospace,monospace;}}
@media (prefers-color-scheme:dark){{:root:not([data-theme=light]){{
  --paper:#161614;--card:#1F1F1C;--ink:#F2F1EC;--ink-2:#C4C3BB;--muted:#8B8A82;
  --rule:#33322D;}}}}
*{{box-sizing:border-box}}
body{{background:var(--paper);color:var(--ink);font-family:var(--f-body);
  margin:0;padding:26px 20px 90px;line-height:1.6}}
.wrap{{max-width:660px;margin:0 auto}}
h1{{font-size:21px;margin:0 0 6px}}
.lede{{color:var(--ink-2);font-size:13.5px;margin:0 0 8px;max-width:60ch}}
.warn{{color:var(--muted);font-size:12px;margin:0 0 22px;max-width:60ch}}
.clip{{background:var(--card);border:1px solid var(--rule);padding:15px 17px;
  margin-bottom:14px;border-radius:3px}}
.clip.done{{opacity:.5}}
.clip header{{display:flex;justify-content:space-between;align-items:center;
  font-size:10px;letter-spacing:.08em;text-transform:uppercase;
  color:var(--muted);margin-bottom:10px}}
audio{{width:100%;margin-bottom:12px}}
dl{{margin:0 0 13px;display:grid;gap:6px}}
dl div{{display:grid;grid-template-columns:150px 1fr;gap:10px;align-items:baseline}}
dt{{font-size:10px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}}
dd{{margin:0;font-family:var(--f-mono);font-size:12px;word-break:break-word}}
dd.want{{color:var(--good)}} dd.heard{{color:var(--ink-2)}}
.vote{{display:flex;gap:7px;flex-wrap:wrap}}
.vote button{{font:inherit;font-size:12.5px;padding:7px 13px;cursor:pointer;
  border:1px solid var(--rule);background:transparent;color:var(--ink);
  border-radius:3px}}
.vote button:hover{{border-color:var(--accent)}}
.vote button.on[data-v=right]{{background:var(--good);color:#fff;border-color:var(--good)}}
.vote button.on[data-v=wrong]{{background:var(--bad);color:#fff;border-color:var(--bad)}}
.vote button.on[data-v=unclear]{{background:var(--muted);color:#fff;border-color:var(--muted)}}
.bar{{position:fixed;left:0;right:0;bottom:0;background:var(--card);
  border-top:1px solid var(--rule);padding:11px 20px;display:flex;gap:13px;
  align-items:center;justify-content:center;font-size:13px}}
.bar button{{font:inherit;font-size:12.5px;padding:7px 15px;cursor:pointer;
  background:var(--accent);color:#fff;border:0;border-radius:3px}}
@media(max-width:560px){{dl div{{grid-template-columns:1fr;gap:2px}}}}
</style></head><body>
<div class="wrap">
  <h1>Listening audit</h1>
  <p class="lede">Play each clip and judge one thing only: did the system say the
  bold green text correctly? Ignore accent, pace and audio quality.{lede_extra}</p>
  <p class="warn">Which system produced each clip is deliberately hidden, and the
  order is shuffled. Verdicts save in this browser as you go.</p>
  {''.join(cards)}
</div>
<div class="bar">
  <span id="prog">0 of {len(rows)} judged</span>
  <button id="copy">Copy verdicts</button>
  <button id="dl">Download verdicts.json</button>
</div>
<script>
// One literal key meant the listening audit, the disagreement audit and
  // every run shared a single slot of browser storage: open a second page
  // and it restored the first page's verdicts against unrelated clip ids.
  var KEY = 'tts-audit-v1:{namespace}';
var saved = {{}};
try {{ saved = JSON.parse(localStorage.getItem(KEY) || '{{}}'); }} catch (e) {{ saved = {{}}; }}

function persist() {{
  try {{ localStorage.setItem(KEY, JSON.stringify(saved)); }} catch (e) {{}}
  var n = Object.keys(saved).length;
  document.getElementById('prog').textContent =
    n + ' of {len(rows)} judged';
}}

document.querySelectorAll('.clip').forEach(function (card) {{
  var key = card.dataset.key;
  card.querySelectorAll('.vote button').forEach(function (b) {{
    if (saved[key] === b.dataset.v) {{ b.classList.add('on'); card.classList.add('done'); }}
    b.addEventListener('click', function () {{
      card.querySelectorAll('.vote button').forEach(function (x) {{ x.classList.remove('on'); }});
      b.classList.add('on');
      card.classList.add('done');
      saved[key] = b.dataset.v;
      persist();
    }});
  }});
}});
persist();

function blob() {{ return JSON.stringify(saved, null, 1); }}
document.getElementById('copy').addEventListener('click', function () {{
  navigator.clipboard.writeText(blob()).then(function () {{
    document.getElementById('copy').textContent = 'Copied';
    setTimeout(function () {{
      document.getElementById('copy').textContent = 'Copy verdicts';
    }}, 1400);
  }});
}});
document.getElementById('dl').addEventListener('click', function () {{
  var a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([blob()], {{type: 'application/json'}}));
  a.download = 'verdicts.json';
  a.click();
}});
</script>
</body></html>"""


def build(run_dir: Path, exclude: frozenset, seed: int) -> Path:
    recs = load_records(run_dir / "results.jsonl")
    if exclude:
        recs = [r for r in recs if r["tts"] not in exclude]
    rows = select(recs, seed)
    out = run_dir / "audit"
    clips = out / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    for r in rows:
        src = run_dir / "audio" / r["audio"]
        if src.exists():
            # Copy under the opaque id, never the cache filename. The cache
            # name is built elsewhere and nothing stops a future adapter from
            # putting the system into it, which would leak through the <audio>
            # src and undo the blinding without any test noticing.
            shutil.copy2(src, clips / f"{page_id(r['key'])}.wav")
    # The key -> stratum map stays OUT of the page and next to it on disk, so
    # scoring can attribute verdicts without the listener ever seeing a label.
    (out / "sample.json").write_text(json.dumps(
        {"seed": seed, "exclude": sorted(exclude),
         "rows": [{"key": r["key"], "id": page_id(r["key"]),
                   "stratum": r["_stratum"],
                   "tts": r["tts"], "category": r["category"],
                   "target": r.get("target"), "hypothesis": r.get("hypothesis"),
                   "written": r["written"]} for r in rows]}, indent=1),
        encoding="utf-8")
    page = out / "audit.html"
    page.write_text(render_page(rows, namespace=f"listen:{run_dir.name}"), encoding="utf-8")
    return page


def report_verdicts(run_dir: Path, verdicts_path: Path) -> int:
    sample = json.loads((run_dir / "audit" / "sample.json").read_text(encoding="utf-8"))
    verdicts = json.loads(verdicts_path.read_text(encoding="utf-8"))
    # Verdicts come back keyed by the page id, not the record key.
    by_key = {r["id"]: r for r in sample["rows"]}
    judged = [(by_key[k], v) for k, v in verdicts.items() if k in by_key]
    if not judged:
        print("no verdicts match this sample", file=sys.stderr)
        return 2

    tally: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row, v in judged:
        tally[row["stratum"]][v] += 1

    print(f"\n{len(judged)} of {len(sample['rows'])} clips judged\n")
    for name, (_want, why) in STRATA.items():
        t = tally.get(name)
        if not t:
            continue
        n = sum(t.values())
        right, wrong = t.get("right", 0), t.get("wrong", 0)
        decided = right + wrong
        print(f"  {name}  ({n} judged)")
        print(f"    said correctly {right} · said wrong {wrong} · unclear "
              f"{t.get('unclear', 0)}")
        if decided:
            print(f"    -> {right / decided:.0%} of decided clips were spoken correctly")
        print(f"    {why}")
        print()

    fm = tally.get("flagged_miss", {})
    decided = fm.get("right", 0) + fm.get("wrong", 0)
    if decided:
        share = fm.get("right", 0) / decided
        print("  VERDICT ON THE ARTIFACT CLAIM")
        if share >= 0.7:
            print(f"    {share:.0%} of flagged failures were spoken correctly. The "
                  f"report is right to discount those categories, and a second\n"
                  f"    recognizer lineage is the fix.")
        elif share <= 0.3:
            print(f"    only {share:.0%} were spoken correctly. Those categories are "
                  f"REAL failures, the dagger is wrong, and the affected\n"
                  f"    systems are being flattered by the current report.")
        else:
            print(f"    {share:.0%} spoken correctly: mixed. Neither reading holds; "
                  f"widen the sample before changing the report.")
        print()

    for name in ("unanimous_fail", "lone_fail"):
        bad = [r for r, v in judged if r["stratum"] == name and v == "right"]
        if bad:
            print(f"  SCORING BUGS from {name} ({len(bad)}): spoken correctly, "
                  f"scored wrong")
            for r in bad[:6]:
                print(f"    {r['category']:<26} want {r['target']!r}")
                print(f"    {'':<26} got  {r['hypothesis']!r}")
            print()
    fp = [r for r, v in judged if r["stratum"] == "false_positive" and v == "wrong"]
    if fp:
        print(f"  FALSE POSITIVES ({len(fp)}): scored correct, spoken wrong. "
              f"These inflate the rates above.")
    ctrl = [r for r, v in judged if r["stratum"] == "unanimous_pass" and v == "wrong"]
    if ctrl:
        print(f"  CONTROL FAILED ({len(ctrl)}): items every system passed were "
              f"spoken wrong. Treat the whole run as suspect.")
    return 0


# =========================================================== disagreement mode
# Two recognizer families scored the same audio. Where they AGREE nothing is in
# question; where they disagree exactly one is wrong, and which one cannot be
# derived from the transcripts -- that is the whole reason a human is here.
#
# Pattern-matching gave two useless answers to "how often does the second
# recognizer repair a broken verbalization": 7% from a hand-picked abbreviation
# list, and 100% from counting items whose target introduces any new word.
# Neither is a rate. The disagreement set needs no detector at all.
DISAGREE_STRATA = {
    "b_passes": (25, "The second recognizer passed where the scorer failed. "
                     "Spoken WRONG here means it repaired a real failure; "
                     "spoken right means the scorer misheard correct speech."),
    "a_passes": (25, "The scorer passed where the second recognizer failed. "
                     "Spoken WRONG here means the scorer is inflating; spoken "
                     "right means the second recognizer misheard."),
}


def select_disagree(rows: list[dict], seed: int = 0) -> list[dict]:
    """Stratified by DIRECTION of disagreement, frozen by seed.

    Direction is the only thing that distinguishes these clips, and it is
    exactly what must not reach the listener: knowing a clip is one the second
    recognizer passed is knowing which answer would be convenient.
    """
    buckets: dict[str, list[dict]] = {k: [] for k in DISAGREE_STRATA}
    for r in rows:
        if r.get("direction") in buckets:
            buckets[r["direction"]].append(r)
    rng = random.Random(seed)
    picked: list[dict] = []
    for name, (want, _why) in DISAGREE_STRATA.items():
        pool = sorted(buckets[name], key=lambda r: r["key"])
        for r in rng.sample(pool, min(want, len(pool))):
            picked.append({**r, "_stratum": name})
    rng.shuffle(picked)
    return picked


def build_disagree(run_dir: Path, disagreement: Path, seed: int) -> Path:
    data = json.loads(disagreement.read_text(encoding="utf-8"))
    rows = select_disagree(data["rows"], seed)
    out = run_dir / "audit-disagree"
    clips = out / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    for r in rows:
        src = run_dir / "audio" / (r.get("audio") or "")
        if src.exists():
            shutil.copy2(src, clips / f"{page_id(r['key'])}.wav")
    (out / "sample.json").write_text(json.dumps({
        "seed": seed, "mode": "disagree", "a": data["a"], "b": data["b"],
        "rows": [{"key": r["key"], "id": page_id(r["key"]),
                  "stratum": r["_stratum"], "tts": r["tts"],
                  "category": r["category"], "target": r.get("target"),
                  "written": r["written"],
                  f"{data['a']}_heard": r.get(f"{data['a']}_heard"),
                  f"{data['b']}_heard": r.get(f"{data['b']}_heard")}
                 for r in rows]}, indent=1), encoding="utf-8")
    page = out / "audit.html"
    # show_heard=False: printing either transcript would hand the listener the
    # answer, since the two recognizers disagree by construction.
    page.write_text(render_page(rows, show_heard=False, namespace=f"disagree:{run_dir.name}"), encoding="utf-8")
    return page


def report_disagree(run_dir: Path, verdicts_path: Path) -> int:
    sample = json.loads((run_dir / "audit-disagree" / "sample.json")
                        .read_text(encoding="utf-8"))
    verdicts = json.loads(verdicts_path.read_text(encoding="utf-8"))
    a, b = sample["a"], sample["b"]
    by_id = {r["id"]: r for r in sample["rows"]}
    judged = [(by_id[k], v) for k, v in verdicts.items()
              if k in by_id and v in ("right", "wrong")]
    if not judged:
        print("no verdicts match this sample", file=sys.stderr)
        return 2

    tally: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row, v in judged:
        tally[row["stratum"]][v] += 1

    print(f"\n{len(judged)} of {len(sample['rows'])} disagreements judged\n")
    bp, ap_ = tally.get("b_passes", {}), tally.get("a_passes", {})

    # A clip the human calls WRONG that a recognizer PASSED is that recognizer
    # scoring a broken verbalization as correct -- the costly error.
    b_repairs, b_n = bp.get("wrong", 0), bp.get("right", 0) + bp.get("wrong", 0)
    a_repairs, a_n = ap_.get("wrong", 0), ap_.get("right", 0) + ap_.get("wrong", 0)

    print(f"  {'':<34}{'judged':>8}{'false pass':>12}{'rate':>8}")
    print(f"  {'-'*34}{'-'*8}{'-'*12}{'-'*8}")
    if b_n:
        print(f"  {b + ' passed, ' + a + ' failed':<34}{b_n:>8}"
              f"{b_repairs:>12}{b_repairs/b_n:>8.0%}")
    if a_n:
        print(f"  {a + ' passed, ' + b + ' failed':<34}{a_n:>8}"
              f"{a_repairs:>12}{a_repairs/a_n:>8.0%}")

    print(f"\n  READ THIS AS:")
    if b_n:
        print(f"    {b_repairs/b_n:.0%} of the clips {b} uniquely passed were spoken WRONG")
        print(f"       -> {b} scored a broken verbalization as correct that often")
        print(f"    {1-b_repairs/b_n:.0%} were spoken right -> {a} misheard correct speech")
    if a_n:
        print(f"    {a_repairs/a_n:.0%} of the clips {a} uniquely passed were spoken WRONG")
        print(f"       -> this is the SCORER's own false-pass rate on contested clips")

    # Scale the measured rate back to the corpus.
    n_dis = sum(1 for _ in sample["rows"])
    print(f"\n  The full disagreement set was reported by disagree.py; applying")
    print(f"  these rates to it converts a guess into a bounded estimate.")
    print(f"  Sample here: {n_dis} clips. Wider is better; this is a floor.")

    unclear = sum(1 for k, v in verdicts.items() if k in by_id and v == "unclear")
    if unclear:
        print(f"\n  {unclear} clip(s) marked unclear and excluded from every rate.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Listening audit for a run")
    ap.add_argument("--run", default="runs/full")
    ap.add_argument("--exclude", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--score", default="", metavar="VERDICTS.JSON")
    ap.add_argument("--disagree", default="", metavar="DISAGREEMENT.JSON",
                    help="build the listening pass over cross-family "
                         "disagreements instead of one lineage's failures")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    run_dir = Path(args.run)

    if args.score:
        # Route by which sample the verdicts belong to.
        if (run_dir / "audit-disagree" / "sample.json").exists() and args.disagree != "":
            return report_disagree(run_dir, Path(args.score))
        if (run_dir / "audit-disagree" / "sample.json").exists() and                 not (run_dir / "audit" / "sample.json").exists():
            return report_disagree(run_dir, Path(args.score))
        return report_verdicts(run_dir, Path(args.score))

    if args.disagree:
        page = build_disagree(run_dir, Path(args.disagree), args.seed)
        n = len(json.loads((run_dir / "audit-disagree" / "sample.json")
                           .read_text(encoding="utf-8"))["rows"])
        print(f"wrote {page}  ({n} clips, seed {args.seed})")
        print("open it, judge every clip, then:")
        print(f"  python audit.py --run {args.run} "
              f"--disagree {args.disagree} "
              f"--score {run_dir / 'audit-disagree' / 'verdicts.json'}")
        return 0

    exclude = frozenset(x.strip() for x in args.exclude.split(",") if x.strip())
    page = build(run_dir, exclude, args.seed)
    n = len(json.loads((run_dir / "audit" / "sample.json")
                       .read_text(encoding="utf-8"))["rows"])
    print(f"wrote {page}  ({n} clips, seed {args.seed})")
    print(f"open it, judge every clip, then:\n"
          f"  python audit.py --run {args.run} --score "
          f"{run_dir / 'audit' / 'verdicts.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
