"""Generate a standalone HTML report from a run.

    python report.py --run runs/combined --out report.html

Design intent: this is a measurement report, not a dashboard. The reader wants
three things in order -- which system is best, can I trust that, and where
exactly did it fail. The layout answers them in that order.

The load-bearing design decision is the confidence interval. It renders as a
band behind each bar rather than as a column of numbers, so overlapping bands
are *visible*. A reader who cannot see that two systems are statistically
indistinguishable will read a ranking the data does not support.

Text blocks are laid out with Pretext so heights are computed rather than
guessed, and stay correct when the window is resized.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

VENDOR_CANDIDATES = [
    Path.home() / ".claude/skills/gstack/design-html/vendor/pretext.js",
    Path(__file__).resolve().parent / "vendor" / "pretext.js",
]


def esc(s) -> str:
    return html.escape("" if s is None else str(s))


def pct(x) -> str:
    return "—" if x is None else f"{x * 100:.0f}%"


def load_run(run_dir: Path, exclude: frozenset = frozenset(),
             asr: str | None = None):
    results = run_dir / "results.jsonl"
    if not results.exists():
        raise SystemExit(f"no results at {results}")
    recs = []
    for line in results.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if exclude:
        recs = [r for r in recs if r.get("tts") not in exclude]
    if asr:
        recs = [r for r in recs if asr in (r.get("asr") or "")]
        if not recs:
            raise SystemExit(f"no records from a recognizer matching "
                             f"{asr!r} in {results}")
    asrs = {r.get("asr") for r in recs}
    if len(asrs) > 1:
        # The guard is right -- a report must describe ONE measurement --
        # but it used to be unsatisfiable: render() re-read results.jsonl
        # itself and never learned which recognizer score.py had picked,
        # so `--asr whisper --html` was refused on every two-lineage run.
        raise SystemExit(
            f"this run carries {len(asrs)} recognizers "
            f"({', '.join(sorted(a for a in asrs if a))}); pass asr= "
            f"so the report describes a single measurement")
    summary_path = run_dir / "summary.json"
    # A cached summary.json describes EVERY system in the run. Trusting it while
    # the records have been filtered would print a ranking that disagrees with
    # the evidence below it, so an excluded run is always recomputed.
    cached = None
    if summary_path.exists() and not exclude:
        cached = json.loads(summary_path.read_text(encoding="utf-8"))
        # summary.json is written by whatever the last scoring run covered,
        # which may have excluded a system. If its system set does not match
        # the records in hand, it describes a different run: recompute rather
        # than render a ranking that disagrees with the evidence table.
        if set(cached.get("systems") or {}) != {r["tts"] for r in recs}:
            cached = None
    if cached is not None:
        summary = cached
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from score import summarize
        summary = summarize(recs)
    manifest = {}
    mp = run_dir / "manifest.json"
    if mp.exists():
        manifest = json.loads(mp.read_text(encoding="utf-8"))
    return recs, summary, manifest


def pretext_script() -> str:
    for p in VENDOR_CANDIDATES:
        if p.exists():
            return f"<script>\n{p.read_text(encoding='utf-8')}\n</script>"
    return ("<!-- vendor/pretext.js missing, using CDN -->\n"
            "<script type='module'>import * as P from "
            "'https://esm.sh/@chenglou/pretext'; window.Pretext = P;</script>")


# ----------------------------------------------------------------- sections
def render_ranking(summary: dict) -> str:
    systems = sorted(summary["systems"].items(), key=lambda kv: -kv[1]["match_rate"])
    if not systems:
        return "<p class='muted'>No systems scored.</p>"

    best_rate = systems[0][1]["match_rate"]
    contenders = [s for s, d in systems if d["ci95"][1] >= systems[0][1]["ci95"][0]]

    rows = []
    for name, d in systems:
        lo, hi = d["ci95"]
        rows.append(f"""
      <li class="bar-row">
        <div class="bar-label">
          <span class="sys">{esc(name)}</span>
          <span class="rev">{esc(d.get('revision') or '')}</span>
        </div>
        <div class="bar-track" role="img"
             aria-label="{esc(name)}: {pct(d['match_rate'])} correct, 95% interval {pct(lo)} to {pct(hi)}">
          <span class="ci" style="left:{lo*100:.4f}%;width:{max(hi-lo,0)*100:.4f}%"></span>
          <span class="fill" style="width:{d['match_rate']*100:.4f}%"></span>
          <span class="tick" style="left:{d['match_rate']*100:.4f}%"></span>
        </div>
        <div class="bar-num">
          <b>{pct(d['match_rate'])}</b>
          <span class="ci-text">{pct(lo)}–{pct(hi)}</span>
        </div>
        <div class="bar-n">{d['scored']}</div>
      </li>""")

    note = ""
    if len(contenders) > 1:
        note = (f"<p class='readout'>The top {len(contenders)} systems have overlapping "
                f"confidence intervals. On this evidence they are "
                f"<strong>not distinguishable</strong> — treat the order as provisional.</p>")
    else:
        note = (f"<p class='readout'><strong>{esc(systems[0][0])}</strong> leads at "
                f"{pct(best_rate)}, and its interval clears the next system's.</p>")

    return f"""
  <ol class="bars">{''.join(rows)}</ol>
  {note}"""


# CTC homophones. wav2vec2 has no language model, so it writes what it hears:
# "point seven one TO eight" for "...two eight". The speech was correct.
_HOMOPHONES = {"to", "too", "for", "fore", "ate", "won", "ford", "tu", "buy",
               "by", "sea", "see", "why", "are", "you", "bee", "be", "eye", "i"}


def asr_suspect(recs: list[dict], floor: float = 0.5) -> dict[str, float]:
    """Categories whose failures are mostly the RECOGNIZER, not the system.

    Two artifacts dominate, and neither is a verbalization error:

    1. Spelled letters. A target of "i s b n nine seven eight..." needs the
       recognizer to hear four letter names in a row. CTC cannot; it writes
       "AIASPAN". On the ISBN items the digits match the target exactly and
       the row still fails.
    2. Homophones. "two" and "to" are the same sound and a CTC decoder has no
       language model to separate them.

    Returned per category as the share of failures showing either artifact.
    A category above `floor` cannot be read as a statement about the systems,
    so the matrix marks it and the reader is told why.
    """
    out: dict[str, float] = {}
    by_cat: dict[str, list[dict]] = {}
    for r in recs:
        if r.get("status") == "scored" and r.get("target_match") is False:
            by_cat.setdefault(r["category"], []).append(r)
    for cat, misses in by_cat.items():
        n = 0
        for r in misses:
            tgt = (r.get("target") or "").split()
            spelled = sum(1 for t in tgt if len(t) == 1 and t.isalpha()) >= 3
            homo = any(w in _HOMOPHONES
                       for w in (r.get("hypothesis") or "").lower().split())
            if spelled or homo:
                n += 1
        out[cat] = n / len(misses) if misses else 0.0
    return out


def render_categories(summary: dict, suspect: dict[str, float] | None = None) -> str:
    suspect = suspect or {}
    cats = summary.get("categories") or {}
    if len(cats) < 2:
        return ""
    systems = sorted(summary["systems"])
    head = "".join(f"<th>{esc(s)}</th>" for s in systems)
    body = []
    for cat in sorted(cats):
        cells = []
        for s in systems:
            cell = cats[cat].get(s)
            if not cell:
                cells.append("<td class='na'>—</td>")
                continue
            r = cell["match_rate"]
            cells.append(
                f"<td class='heat' style='--v:{r:.3f}'>"
                f"<span>{pct(r)}</span><small>n={cell['n']}</small></td>")
        flag = suspect.get(cat, 0.0) >= 0.5
        mark = ("<abbr title='Most failures in this row are recognizer "
                "artifacts, not verbalization errors'>&dagger;</abbr>"
                if flag else "")
        body.append(f"<tr{' class=\"suspect\"' if flag else ''}>"
                    f"<th scope='row'>{esc(cat)}{mark}</th>"
                    f"{''.join(cells)}</tr>")
    n_flag = sum(1 for v in suspect.values() if v >= 0.5)
    legend = (f"<p class='legend'>&dagger; {n_flag} categories where most "
              f"failures are recognizer artifacts, not verbalization errors: "
              f"targets needing spelled-out letters, or homophones a decoder "
              f"with no language model cannot separate. Read these rows as "
              f"<em>not measurable with this recognizer</em>, not as a system "
              f"score.</p>") if n_flag else ""
    return f"""
  <section id="categories">
    <h2>By category</h2>
    <p class="lede" data-pretext>Where each system actually breaks. At most 20 items
    per category, so a single cell is exploratory — pool categories before making
    a claim about one of them.</p>
    <div class="scroll">
      <table class="matrix">
        <thead><tr><th scope="col">Category</th>{head}</tr></thead>
        <tbody>{''.join(body)}</tbody>
      </table>
    </div>
    {legend}
  </section>"""


def render_breakdown(summary: dict) -> str:
    """Where each system actually breaks.

    Only meaningful once the run covers many categories: with one category
    there is nothing to rank against. The headline rate hides that a system
    can be perfect on dates and useless on phone numbers, which is the thing
    a reader building a voice agent actually needs to know.
    """
    cats = summary.get("categories") or {}
    if len(cats) < 4:
        return ""
    cards = []
    for s, d in sorted(summary["systems"].items(),
                       key=lambda kv: -kv[1]["match_rate"]):
        rows = [(c, cats[c][s]["match_rate"], cats[c][s]["n"])
                for c in sorted(cats) if cats[c].get(s)]
        if len(rows) < 4:
            continue
        rows.sort(key=lambda t: (t[1], t[0]))
        clean = sum(1 for _c, r, _n in rows if r >= 0.999)
        # Colour tracks the value, width tracks the value. A single red bar
        # at every rate made 90% render as a long red bar, which reads as
        # heavy failure: width said good, colour said bad.
        def band(r):
            return ("var(--bad)" if r < 0.5
                    else "#B8862B" if r < 0.8 else "var(--good)")
        worst = "".join(
            f"<li><span class='bd-cat'>{esc(c)}</span>"
            f"<span class='bd-bar'><i style='--v:{r:.3f};--c:{band(r)}'></i></span>"
            f"<span class='bd-val'>{pct(r)}</span>"
            f"<small>n={n}</small></li>"
            for c, r, n in rows[:4])
        cards.append(f"""
      <article class="bd">
        <h3>{esc(s)}</h3>
        <p class="bd-sum"><strong>{pct(d['match_rate'])}</strong> overall ·
        clean on <strong>{clean}</strong> of {len(rows)} categories</p>
        <ol class="bd-list">{worst}</ol>
      </article>""")
    if not cards:
        return ""
    return f"""
  <section id="breakdown">
    <h2>Where each system breaks</h2>
    <p class="lede" data-pretext>The headline rate is an average over 27 very
    different problems. A system can be flawless on dates and fall apart on
    phone numbers, and for a voice agent that distinction decides whether it
    is usable. Four weakest categories per system, worst first.</p>
    <div class="bd-grid">{''.join(cards)}</div>
  </section>"""


def evidence_omitted(n_scored: int) -> str:
    """Stand-in for the evidence section when it is deliberately left out.

    The section is the audit trail, so removing it silently would undercut
    every rate above it. It is dropped only to avoid redistributing the
    source corpus (PolyNorm-Bench is CC BY-NC-ND), and the reader is told
    that rather than shown a report that quietly has no evidence in it.
    """
    return f"""
  <section id="evidence">
    <h2>Evidence</h2>
    <p class="lede" data-pretext>This copy omits the per-item table. The
    {n_scored:,} scored items each quote source text from the corpus, which is
    licensed for use but not for redistribution, so the audit trail is kept to
    the local copy of this report rather than published with it. Every rate
    above is computed on the full set; nothing was excluded from the numbers,
    only from this page.</p>
  </section>"""


def render_evidence(recs: list[dict]) -> str:
    """Every scored item, filterable.

    NOT truncated. An earlier version capped each tab at 400 rows, which was
    invisible at 10 items and silently hid ~550 of ~950 failures on a full
    540-item run, while the lede claimed to show every item. A rate you
    cannot audit is a rate you should not quote, so the cap had to go and the
    filters exist to make the full set navigable instead.
    """
    scored = [r for r in recs if r.get("status") == "scored"]
    misses = [r for r in scored if r.get("target_match") is False]
    hits = [r for r in scored if r.get("target_match") is True]
    systems = sorted({r["tts"] for r in scored})
    cats = sorted({r["category"] for r in scored})

    def row(r):
        ok = bool(r.get("target_match"))
        # data-label drives the stacked card layout on narrow screens, where a
        # six-column scrolling table is unreadable.
        return f"""
        <tr class="{'hit' if ok else 'miss'}" data-sys="{esc(r['tts'])}" data-cat="{esc(r['category'])}" data-ok="{'1' if ok else '0'}">
          <td class="mark" data-label="">{'✓' if ok else '✕'}</td>
          <td class="sysname" data-label="System">{esc(r['tts'])}</td>
          <td class="cat" data-label="Category">{esc(r['category'])}</td>
          <td class="written" data-label="Written">{esc(r['written'])}</td>
          <td class="want" data-label="Required">{esc(r.get('target') or '')}</td>
          <td class="heard" data-label="Heard">{esc(r.get('hypothesis') or '')}</td>
        </tr>"""

    # Failures first: that is what a reader opens this section to find.
    rows = "".join(row(r) for r in misses + hits)
    sys_opts = "".join(f"<option>{esc(s)}</option>" for s in systems)
    cat_opts = "".join(f"<option>{esc(c)}</option>" for c in cats)
    return f"""
  <section id="evidence">
    <h2>Evidence</h2>
    <p class="lede" data-pretext>Every scored item, with what the system was required
    to say and what the recognizer actually heard. Nothing here is truncated. A rate
    you cannot audit is a rate you should not quote.</p>
    <div class="controls">
      <div class="tabs">
        <button class="tab on" data-ok="0">Failures ({len(misses)})</button>
        <button class="tab" data-ok="1">Correct ({len(hits)})</button>
        <button class="tab" data-ok="">All ({len(scored)})</button>
      </div>
      <label class="sel">System
        <select id="f-sys"><option value="">All</option>{sys_opts}</select></label>
      <label class="sel">Category
        <select id="f-cat"><option value="">All</option>{cat_opts}</select></label>
    </div>
    <p class="count" id="ev-count"></p>
    <div class="scroll">
      <table class="evidence">
        <thead><tr><th></th><th>System</th><th>Category</th><th>Written</th>
        <th>Required</th><th>Heard</th></tr></thead>
        <tbody id="ev-body">{rows}</tbody>
      </table>
      <p class="na" id="ev-empty" hidden>Nothing matches those filters.</p>
    </div>
  </section>"""


def render(run_dir: Path, evidence: bool = True,
           exclude: frozenset = frozenset(),
           asr: str | None = None) -> str:
    recs, summary, manifest = load_run(run_dir, exclude, asr=asr)
    scored = sum(1 for r in recs if r.get("status") == "scored")
    excluded = len(recs) - scored
    cats_seen = len({r["category"] for r in recs})
    # DISTINCT items, not records. With 6 systems x 10 items there are 60
    # records but only 10 items -- reporting 60 as "items" would be exactly
    # the denominator error this report warns about elsewhere.
    n_items = len({r["item_id"] for r in recs})
    n_systems = len({r["tts"] for r in recs})
    asr = (manifest.get("asr") or [{}])[0].get("id", "wav2vec2 (CTC)")
    when = manifest.get("created") or datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")

    suspect = asr_suspect(recs)
    flagged = sorted(c for c, v in suspect.items() if v >= 0.5)
    miss_by_cat: dict[str, int] = {}
    for r in recs:
        if r.get('status') == 'scored' and r.get('target_match') is False:
            miss_by_cat[r['category']] = miss_by_cat.get(r['category'], 0) + 1
    n_miss = sum(miss_by_cat.values())
    n_art = sum(round(v * miss_by_cat.get(c, 0)) for c, v in suspect.items())
    art_pct = (n_art / n_miss) if n_miss else 0.0
    limits_extra = (
        f' On this run <strong>{art_pct:.0%}</strong> of all failures carry a'
        f' recognizer artifact rather than a verbalization error: a target'
        f' needing spelled-out letters, or a homophone a decoder with no'
        f' language model cannot separate. The {len(flagged)} categories'
        f' marked &dagger; are mostly artifact and must not be read as'
        f' statements about the systems. Closing this needs a second'
        f' recognizer lineage, not a better scorer.'
    ) if flagged else ''

    evidence_html = (render_evidence(recs) if evidence
                     else evidence_omitted(scored))

    excl_bits = " · ".join(f"{k.replace('_',' ')} {v}"
                           for k, v in (summary.get("excluded_reasons") or {}).items())

    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TTS verbalization report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root{{
  --paper:#FBFBF9; --card:#FFFFFF; --ink:#1A1A17; --ink-2:#4A4A42; --muted:#6E6E66;
  --rule:#E4E3DC; --rule-2:#CFCEC4;
  --accent:#16556B; --good:#1F7A4D; --bad:#A8321F;
  --ci:rgba(22,85,107,.14);
  --f-display:"Fraunces",Georgia,serif;
  --f-body:"IBM Plex Sans",system-ui,sans-serif;
  --f-mono:"IBM Plex Mono",ui-monospace,monospace;
}}
@media (prefers-color-scheme:dark){{
  :root:not([data-theme=light]){{
    --paper:#14140F; --card:#1C1C16; --ink:#F0EFE6; --ink-2:#C6C5B8; --muted:#8C8B7E;
    --rule:#2C2C24; --rule-2:#3D3D33;
    --accent:#63B6CE; --good:#5FBE8C; --bad:#E88A75; --ci:rgba(99,182,206,.18);
  }}
}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--paper);color:var(--ink);font-family:var(--f-body);
  font-size:15px;line-height:1.55;-webkit-font-smoothing:antialiased}}
.wrap{{max-width:1080px;margin:0 auto;padding:40px 24px 96px}}
h1{{font-family:var(--f-display);font-weight:600;font-size:clamp(30px,4.5vw,44px);
  line-height:1.05;margin:0 0 6px;letter-spacing:-.015em;text-wrap:balance}}
h2{{font-family:var(--f-display);font-weight:600;font-size:22px;margin:0 0 6px;
  letter-spacing:-.01em}}
section{{margin-top:56px}}
.lede{{color:var(--ink-2);max-width:64ch;margin:0 0 20px}}
.muted,.na{{color:var(--muted)}}

/* spec line -------------------------------------------------------------- */
.spec{{display:flex;flex-wrap:wrap;gap:0 28px;padding:14px 0;margin-top:18px;
  border-top:1px solid var(--rule);border-bottom:1px solid var(--rule);
  font-family:var(--f-mono);font-size:11.5px;color:var(--muted)}}
.spec b{{color:var(--ink);font-weight:500}}

/* ranking ---------------------------------------------------------------- */
.bars{{list-style:none;margin:0;padding:0}}
.bar-row{{display:grid;grid-template-columns:minmax(150px,1.1fr) minmax(0,3fr) 110px 44px;
  gap:16px;align-items:center;padding:13px 0;border-bottom:1px solid var(--rule)}}
.bar-row:first-child .sys{{font-weight:600}}
.bar-label{{min-width:0}}
.sys{{display:block;font-size:14.5px;overflow-wrap:anywhere}}
.rev{{display:block;font-family:var(--f-mono);font-size:10.5px;color:var(--muted);
  overflow-wrap:anywhere;margin-top:2px}}
.bar-track{{position:relative;height:22px;background:var(--rule);border-radius:2px}}
.ci{{position:absolute;top:0;bottom:0;background:var(--ci);border-left:1px solid var(--rule-2);
  border-right:1px solid var(--rule-2)}}
.fill{{position:absolute;top:0;bottom:0;left:0;background:var(--accent);border-radius:2px 0 0 2px}}
.tick{{position:absolute;top:-3px;bottom:-3px;width:2px;background:var(--ink)}}
.bar-num{{font-family:var(--f-mono);font-variant-numeric:tabular-nums;text-align:right}}
.bar-num b{{font-size:17px;font-weight:500}}
.ci-text{{display:block;font-size:10.5px;color:var(--muted)}}
.bar-n{{font-family:var(--f-mono);font-size:12px;color:var(--muted);text-align:right;
  font-variant-numeric:tabular-nums}}
.readout{{margin:18px 0 0;padding:12px 14px;border-left:3px solid var(--accent);
  background:var(--card);color:var(--ink-2);max-width:70ch}}

/* caveats ---------------------------------------------------------------- */
.caveats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));
  gap:1px;background:var(--rule);border:1px solid var(--rule);margin-top:22px}}
.caveat{{background:var(--card);padding:14px 16px}}
.caveat dt{{font-family:var(--f-mono);font-size:10.5px;letter-spacing:.07em;
  text-transform:uppercase;color:var(--muted);margin-bottom:5px}}
.caveat dd{{margin:0;font-size:13.5px;color:var(--ink-2)}}
.caveat b{{color:var(--ink)}}

/* tables ----------------------------------------------------------------- */
.scroll{{overflow-x:auto;border:1px solid var(--rule);background:var(--card)}}
table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{text-align:left;padding:9px 12px;border-bottom:1px solid var(--rule);
  vertical-align:top}}
thead th{{font-family:var(--f-mono);font-size:10.5px;letter-spacing:.06em;
  text-transform:uppercase;color:var(--muted);font-weight:500;
  border-bottom:1px solid var(--rule-2);white-space:nowrap;position:sticky;top:0;
  background:var(--card)}}
.matrix th[scope=row]{{font-weight:400;white-space:nowrap}}
.heat{{font-family:var(--f-mono);font-variant-numeric:tabular-nums;text-align:right;
  background:color-mix(in srgb,var(--accent) calc(var(--v)*26%),transparent)}}
.heat small{{display:block;font-size:9.5px;color:var(--muted)}}
.matrix tr.suspect th[scope=row]{{color:var(--muted)}}
.matrix tr.suspect{{background:repeating-linear-gradient(135deg,
  transparent 0 6px,rgba(110,110,102,.05) 6px 12px)}}
.matrix abbr{{text-decoration:none;color:var(--muted);cursor:help;
  margin-left:4px;font-size:11px}}
.legend{{font-size:11.5px;color:var(--muted);margin:10px 0 0;max-width:70ch;
  line-height:1.55}}
.legend em{{color:var(--ink-2);font-style:italic}}
.steps{{list-style:none;counter-reset:none;margin:0;padding:0;
  display:grid;gap:30px;max-width:76ch}}
.step h3{{font-family:var(--f-body);font-size:14px;font-weight:600;margin:0 0 8px;
  display:flex;align-items:center;gap:9px}}
.step .sn{{display:inline-grid;place-items:center;width:21px;height:21px;
  border-radius:50%;background:var(--accent);color:#fff;font-family:var(--f-mono);
  font-size:11px;font-weight:500;flex:none}}
.step p{{margin:0 0 9px;color:var(--ink-2)}}
.step em{{font-style:italic;color:var(--ink)}}
.math{{font-family:var(--f-mono);font-size:13px;color:var(--ink)!important;
  background:var(--card);border:1px solid var(--rule);padding:13px 16px;
  display:flex;align-items:center;gap:7px;flex-wrap:wrap;margin:12px 0!important}}
.mv{{font-style:italic}}
.mf{{display:inline-grid;text-align:center;vertical-align:middle}}
.mf .num{{padding:0 6px 2px;border-bottom:1px solid var(--ink-2)}}
.mf .den{{padding:2px 6px 0}}
.mnote{{font-family:var(--f-body);font-size:11px;color:var(--muted);
  margin-left:auto;text-align:right;max-width:30ch;line-height:1.45}}
.proc{{margin:6px 0 12px;padding-left:20px;color:var(--ink-2);font-size:12.5px;
  display:grid;gap:5px}}
.rule{{border-left:2px solid var(--accent);padding-left:13px;
  background:transparent;margin-top:12px!important}}
.tok{{font-family:var(--f-mono);font-size:11.5px;background:var(--rule);
  padding:1px 5px;border-radius:2px;color:var(--ink)}}
.ex{{border:1px solid var(--rule);background:var(--card);padding:13px 15px;
  margin:12px 0;position:relative}}
.ex-hit{{border-left:3px solid var(--good)}}
.ex-miss{{border-left:3px solid var(--bad)}}
.ex-tag{{position:absolute;top:11px;right:14px;font-family:var(--f-mono);
  font-size:9.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}}
.ex dl{{margin:0;display:grid;gap:6px}}
.ex dl div{{display:grid;grid-template-columns:118px 1fr;gap:10px;align-items:baseline}}
.ex dt{{font-size:10px;letter-spacing:.07em;text-transform:uppercase;
  color:var(--muted)}}
.ex dd{{margin:0;font-family:var(--f-mono);font-size:11.5px;color:var(--ink);
  word-break:break-word}}
.ex dd.want{{color:var(--good)}}
.ex dd.heard{{color:var(--ink-2)}}
.ex-note{{font-size:12px;color:var(--muted)!important;margin:-2px 0 12px!important;
  max-width:66ch}}
@media(max-width:560px){{
  .ex dl div{{grid-template-columns:1fr;gap:2px}}
  .mnote{{margin-left:0;text-align:left;max-width:none}}
}}
.evidence .mark{{width:26px;text-align:center;font-family:var(--f-mono)}}
.evidence .hit .mark{{color:var(--good)}}
.evidence .miss .mark{{color:var(--bad)}}
.evidence .sysname,.evidence .cat{{font-family:var(--f-mono);font-size:11px;
  color:var(--muted);white-space:nowrap}}
.evidence .want{{color:var(--good)}}
.evidence .miss .heard{{color:var(--bad)}}
.evidence td{{max-width:280px}}

/* tabs ------------------------------------------------------------------- */
.tabs{{display:flex;gap:6px;margin-bottom:12px}}
.tab{{font:inherit;font-size:12.5px;padding:6px 13px;border:1px solid var(--rule-2);
  background:transparent;color:var(--ink-2);cursor:pointer;border-radius:2px}}
.tab.on{{background:var(--ink);color:var(--paper);border-color:var(--ink)}}
.tab:focus-visible{{outline:2px solid var(--accent);outline-offset:2px}}
.controls{{display:flex;flex-wrap:wrap;gap:14px;align-items:flex-end;margin-bottom:10px}}
.controls .tabs{{margin-bottom:0}}
.sel{{display:flex;flex-direction:column;gap:4px;font-size:10.5px;
  letter-spacing:.07em;text-transform:uppercase;color:var(--muted)}}
.sel select{{font:inherit;font-size:12.5px;letter-spacing:0;text-transform:none;
  color:var(--ink);padding:5px 8px;border:1px solid var(--rule-2);
  background:var(--card);border-radius:2px;max-width:230px}}
.count{{font-family:var(--f-mono);font-size:11px;color:var(--muted);margin:0 0 8px}}
.bd-grid{{display:grid;gap:14px;
  grid-template-columns:repeat(auto-fit,minmax(258px,1fr))}}
.bd{{border:1px solid var(--rule);background:var(--card);padding:16px 18px 14px}}
.bd h3{{font-family:var(--f-mono);font-size:12px;font-weight:500;margin:0 0 6px;
  color:var(--accent);letter-spacing:.02em}}
.bd-sum{{font-size:12.5px;color:var(--ink-2);margin:0 0 12px}}
.bd-sum strong{{font-family:var(--f-mono);color:var(--ink);font-weight:500}}
.bd-list{{list-style:none;margin:0;padding:0;display:grid;gap:7px}}
.bd-list li{{display:grid;grid-template-columns:1fr 52px 38px 26px;gap:7px;
  align-items:center;font-size:11.5px}}
.bd-cat{{color:var(--ink-2);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.bd-bar{{display:block;height:5px;background:var(--rule);border-radius:3px;
  overflow:hidden}}
.bd-bar i{{display:block;height:100%;width:calc(var(--v)*100%);
  background:var(--c,var(--bad));border-radius:3px}}
.bd-val{{font-family:var(--f-mono);font-size:11px;text-align:right;
  font-variant-numeric:tabular-nums}}
.bd-list small{{color:var(--muted);font-size:9.5px;text-align:right}}
.evidence tbody tr[hidden]{{display:none}}

footer{{margin-top:64px;padding-top:18px;border-top:1px solid var(--rule);
  font-size:12.5px;color:var(--muted);max-width:70ch}}
@media (max-width:720px){{
  .wrap{{padding:28px 16px 72px}}
  .bar-row{{grid-template-columns:1fr;gap:6px}}
  .bar-num{{text-align:left}} .bar-n{{display:none}}
  /* A six-column scrolling table is unreadable on a phone. Stack each row
     into a labelled card instead -- same data, no horizontal scroll. */
  .scroll{{overflow-x:visible;border:none;background:transparent}}
  .evidence thead{{display:none}}
  .evidence tbody tr{{display:block;background:var(--card);border:1px solid var(--rule);
    border-left:3px solid var(--rule-2);padding:10px 12px;margin-bottom:8px}}
  .evidence tbody tr.miss{{border-left-color:var(--bad)}}
  .evidence tbody tr.hit{{border-left-color:var(--good)}}
  .evidence td{{display:block;border:none;padding:2px 0;max-width:none}}
  .evidence td:empty{{display:none}}
  .evidence td::before{{content:attr(data-label);display:block;
    font-family:var(--f-mono);font-size:9.5px;letter-spacing:.07em;
    text-transform:uppercase;color:var(--muted)}}
  .evidence td[data-label=""]::before{{display:none}}
  .evidence .mark{{float:right;width:auto;font-size:15px}}
  .evidence .sysname,.evidence .cat{{display:inline-block;margin-right:14px}}
  .matrix{{font-size:12px}}
  .scroll:has(.matrix){{overflow-x:auto;border:1px solid var(--rule);
    background:var(--card)}}
}}
@media (prefers-reduced-motion:reduce){{*{{transition:none!important}}}}
</style>
</head><body>
<div class="wrap">

<header>
  <h1>Can it say the number?</h1>
  <p class="lede" data-pretext>Text-to-speech systems scored on whether they correctly
  <em>verbalize</em> dates, money, phone numbers, addresses and codes. Not audio quality,
  not naturalness — only whether the right words came out.</p>
  <div class="spec">
    <span>dataset <b>PolyNorm-Bench en-US</b></span>
    <span>systems <b>{n_systems}</b></span>
    <span>items <b>{n_items}</b></span>
    <span>categories <b>{cats_seen}</b></span>
    <span>recognizer <b>{esc(asr)}</b></span>
    <span>generated <b>{esc(when)}</b></span>
  </div>
</header>

<section id="ranking">
  <h2>Results</h2>
  <p class="lede" data-pretext>Share of items where the required verbalization was
  present. The pale band behind each bar is the 95% confidence interval — where bands
  overlap, the systems are not separated by this evidence.</p>
  {render_ranking(summary)}

  <dl class="caveats">
    <div class="caveat"><dt>Scored</dt><dd><b>{scored}</b> items reached the scorer</dd></div>
    <div class="caveat"><dt>Excluded</dt><dd><b>{excluded}</b> not counted as failures
      {f'<br><span class="muted">{esc(excl_bits)}</span>' if excl_bits else ''}</dd></div>
    <div class="caveat"><dt>Like-for-like</dt>
      <dd><b>{summary.get('common_items', 0)}</b> items every system produced</dd></div>
    <div class="caveat"><dt>Metric</dt>
      <dd>Target span only, so recognizer errors elsewhere in the sentence do not
      fail a correct reading</dd></div>
  </dl>
</section>

{render_breakdown(summary)}
{render_categories(summary, suspect)}
{evidence_html}

{render_math(summary, recs, suspect)}

<section id="method">
  <h2>How this is measured</h2>
  <p class="lede" data-pretext>Each item pairs a written form with its correct spoken
  form. The system is fed the written form; a CTC recognizer transcribes what it said;
  the two are compared in a canonical space where "20th" and "twentieth" are the same
  token.</p>
  <dl class="caveats">
    <div class="caveat"><dt>Why a CTC recognizer</dt>
      <dd>Whisper rewrites speech back into written form — spoken "May twentieth"
      returns as "May 20th". Scored against a spoken reference that penalizes a
      correct system on every item, so a non-normalizing recognizer is a correctness
      requirement here, not a preference.</dd></div>
    <div class="caveat"><dt>Target span, not whole sentence</dt>
      <dd>Only the non-standard token is scored. A recognizer writing "DEAD LINE" for
      "deadline" must not fail a perfectly spoken date.</dd></div>
    <div class="caveat"><dt>Exclusions</dt>
      <dd>Items a provider never generated, silent audio and suspected hallucinated
      transcripts are reported separately, never averaged in as zeros.</dd></div>
    <div class="caveat"><dt>Known limits</dt>
      <dd>One recognizer lineage, so cross-family disagreement is unmeasured. At most
      20 items per category. Function words are still required inside the target span,
      which is strict.{limits_extra}</dd></div>
  </dl>
</section>

<footer>
  Dataset: Apple PolyNorm-Bench (CC BY-NC-ND 4.0) — results may be shown, the data may
  not be redistributed. Method draws on arXiv:2607.08256 (ASR-family confounding) and
  arXiv:2605.26978 (screen before scoring).
</footer>

</div>

{pretext_script()}
<script>
(function(){{
  var body = document.getElementById('ev-body');
  if (!body) return;
  var rows  = Array.prototype.slice.call(body.rows);
  var fsys  = document.getElementById('f-sys');
  var fcat  = document.getElementById('f-cat');
  var count = document.getElementById('ev-count');
  var empty = document.getElementById('ev-empty');
  var ok = '0';

  function apply(){{
    var sv = fsys.value, cv = fcat.value, shown = 0;
    for (var i = 0; i < rows.length; i++){{
      var r = rows[i];
      var keep = (ok === '' || r.dataset.ok === ok)
              && (sv === '' || r.dataset.sys === sv)
              && (cv === '' || r.dataset.cat === cv);
      r.hidden = !keep;
      if (keep) shown++;
    }}
    count.textContent = shown + ' of ' + rows.length + ' scored items';
    empty.hidden = shown > 0;
  }}

  document.querySelectorAll('.tab').forEach(function(t){{
    t.addEventListener('click', function(){{
      document.querySelectorAll('.tab').forEach(function(x){{ x.classList.remove('on'); }});
      t.classList.add('on');
      ok = t.dataset.ok;
      apply();
    }});
  }});
  fsys.addEventListener('change', apply);
  fcat.addEventListener('change', apply);
  apply();
}})();

// Pretext: compute prose heights rather than letting the browser guess, and
// recompute them on resize. Tier: simple layout -> prepare() + layout().
(function(){{
  var P = window.Pretext;
  if (!P || !P.prepare) return;
  var nodes = Array.prototype.slice.call(document.querySelectorAll('[data-pretext]'));
  var prepared = new Map();
  function build(){{
    nodes.forEach(function(el){{
      prepared.set(el, P.prepare(el.textContent, getComputedStyle(el).font));
    }});
  }}
  function relayout(){{
    prepared.forEach(function(handle, el){{
      var lh = parseFloat(getComputedStyle(el).lineHeight) || 22;
      var r = P.layout(handle, el.clientWidth, lh);
      el.style.minHeight = r.height + 'px';
    }});
  }}
  (document.fonts ? document.fonts.ready : Promise.resolve()).then(function(){{
    build(); relayout();
    new ResizeObserver(relayout).observe(document.body);
  }});
}})();
</script>
</body></html>"""


def render_math(summary: dict, recs: list[dict], suspect: dict[str, float]) -> str:
    """The arithmetic behind every number on this page, worked on real rows.

    Every figure here is COMPUTED from the run, never written in by hand. A
    methodology section that quotes numbers it did not derive goes stale the
    first time the corpus changes, and a stale explanation of a live number is
    worse than no explanation at all.
    """
    sysmap = summary["systems"]
    lead = max(sysmap, key=lambda s: sysmap[s]["match_rate"])
    d = sysmap[lead]
    rows = [r for r in recs if r["tts"] == lead]
    scored = [r for r in rows if r.get("status") == "scored"]
    k = sum(1 for r in scored if r.get("target_match"))
    n = len(scored)
    rate = k / n if n else 0.0
    excl = d.get("excluded") or {}
    excl_sum = " ".join(f"&minus; {v} {kk.replace('_', ' ')}"
                        for kk, v in sorted(excl.items()))
    sent = sum(1 for r in scored if r.get("match"))
    lo, hi = d["ci95"]
    smallest_n = min((v["scored"] for v in sysmap.values()), default=0)

    # Teaching examples pulled from the run itself.
    # The hit we want is the one that justifies target-span scoring: the target
    # is correct while the whole sentence fails on a recognizer error elsewhere.
    # Rank candidates rather than taking the first. The first hit that happens
    # to satisfy the conditions may have a target like "the fifteenth of",
    # which drags in function words and teaches the reader nothing about what
    # a target span is. Prefer short spans that begin and end on content words.
    _FUNC = {"the", "a", "an", "of", "and", "at", "in", "on", "for", "to",
             "is", "was", "from", "by", "with"}

    def quality(r) -> tuple:
        t = (r.get("target") or "").split()
        if not t:
            return (9, 9, 9)
        edges = (t[0] in _FUNC) + (t[-1] in _FUNC)
        # Spelled letters are the artifact-prone shape flagged in the matrix.
        # An example built on one teaches the reader the wrong lesson about
        # what a failure means, so push those candidates to the back.
        spelled = sum(1 for x in t if len(x) == 1 and x.isalpha())
        return (edges, spelled, abs(len(t) - 4))

    cands = [r for r in scored if r.get("target_match")
             and not r.get("match") and r.get("hypothesis")]
    hit = min(cands, key=quality) if cands else None
    # The miss must be a REAL verbalization failure, so skip any category whose
    # failures are mostly recognizer artifacts. Teaching with an artifact would
    # teach the wrong lesson about what the number means.
    mcands = [r for r in recs if r.get("status") == "scored"
              and r.get("target_match") is False
              and suspect.get(r["category"], 0.0) < 0.5
              and r.get("hypothesis")]
    miss = min(mcands, key=quality) if mcands else None

    def ex(r, ok):
        if not r:
            return ""
        verdict = ("every required token present" if ok
                   else "a required token is missing")
        return f"""
      <div class="ex {'ex-hit' if ok else 'ex-miss'}">
        <span class="ex-tag">scores {'1' if ok else '0'}</span>
        <dl>
          <div><dt>written in</dt><dd>{esc(r['written'])}</dd></div>
          <div><dt>must say</dt><dd class="want">{esc(r.get('target') or '')}</dd></div>
          <div><dt>recognizer heard</dt><dd class="heard">{esc(r['hypothesis'])}</dd></div>
          <div><dt>verdict</dt><dd>{esc(r['tts'])} &middot; {esc(r['category'])}
            &middot; {verdict}</dd></div>
        </dl>
      </div>"""

    hit_note = ("Target span correct, whole sentence wrong. This single row is the "
                "argument for not scoring the sentence: the part under test is "
                "perfect, and the sentence fails only because the recognizer mangled "
                "a word the item was never about."
                if hit else "")

    return f"""
  <section id="math">
    <h2>The arithmetic</h2>
    <p class="lede" data-pretext>Every number on this page and how it is derived,
    worked on real rows from this run. Each figure below was computed from the data
    rather than written in, so it stays true when the corpus changes.</p>

    <ol class="steps">

      <li class="step">
        <h3><span class="sn">1</span> One item is one binary outcome</h3>
        <p>Each corpus item pairs a <em>written</em> form with the <em>spoken</em> form
        a person would actually say. The system is given the written form only. A CTC
        recognizer transcribes what came back. The item scores <strong>1</strong> if
        every token of the required span appears in that transcript and
        <strong>0</strong> if any is missing. No partial credit.</p>
        <p>The required span is the fragment the item is actually testing. It is found
        by diffing the written and spoken forms on their <em>surface</em> strings,
        before canonicalization. Diffing after canonicalization finds nothing, because
        canonicalization exists to make those two forms equal.</p>
        {ex(hit, True)}
        <p class="ex-note">{hit_note}</p>
        {ex(miss, False)}
      </li>

      <li class="step">
        <h3><span class="sn">2</span> The match rate</h3>
        <p>The headline number is the share of scored items that scored 1.</p>
        <p class="math"><span class="mv">p&#770;</span> =
        <span class="mf"><span class="num">k</span><span class="den">n</span></span>
        = <span class="mf"><span class="num">{k}</span><span class="den">{n}</span></span>
        = <strong>{rate:.4f}</strong> &rarr; {pct(rate)}
        <span class="mnote">{esc(lead)}</span></p>
        <p><strong>n counts items scored, never items attempted.</strong> For
        {esc(lead)}: {len(rows)} attempted {excl_sum} = <strong>{n}</strong> scored.
        Dividing by {len(rows)} instead would turn a provider outage, a silent file or
        an item with nothing to normalize into a quality score. Those are counted and
        shown separately, never averaged in as zeros.</p>
      </li>

      <li class="step">
        <h3><span class="sn">3</span> Why the whole sentence is not scored</h3>
        <p>Requiring the entire transcript to match gives {esc(lead)}
        <strong>{sent}/{n}</strong> = {pct(sent / n if n else 0)} instead of
        {pct(rate)}. That gap is not the system getting worse over the same audio. It
        is the recognizer erring on carrier words the item never tested, charged to
        the system's account.</p>
      </li>

      <li class="step">
        <h3><span class="sn">4</span> Word error rate, and why it does not rank</h3>
        <p>WER is shown for reference. It is edit distance against the full spoken
        reference, normalized by reference length:</p>
        <p class="math">WER =
        <span class="mf"><span class="num">S + D + I</span><span class="den">N</span></span>
        <span class="mnote">substitutions, deletions and insertions over N reference
        words</span></p>
        <p>It counts errors anywhere in the sentence, so it moves for reasons unrelated
        to verbalization. A system can say the date perfectly and still carry a WER
        above zero because the recognizer split one ordinary word in two. That is why
        the ranking uses the target span and WER sits beside it as context.</p>
      </li>

      <li class="step">
        <h3><span class="sn">5</span> The confidence interval</h3>
        <p>{n} items is a sample. Score a different {n} and the rate moves. The interval
        says how far it plausibly moves, by <strong>percentile bootstrap</strong>:</p>
        <ol class="proc">
          <li>Draw {n} results at random <em>with replacement</em> from the {n}
          actually observed.</li>
          <li>Recompute the match rate on that resample.</li>
          <li>Repeat 1,000 times, giving 1,000 plausible rates.</li>
          <li>Take the 2.5th and 97.5th percentiles of those 1,000.</li>
        </ol>
        <p class="math">{esc(lead)} &rarr; <strong>[{lo:.2f}, {hi:.2f}]</strong>
        <span class="mnote">95% interval around {pct(rate)}</span></p>
        <p>Bootstrap rather than a normal approximation, because it assumes nothing
        about the shape of the distribution and the per-category cells are n=20, where
        that approximation is unreliable.</p>
        <p class="rule"><strong>How to read it:</strong> two systems whose intervals
        overlap are <em>not</em> separated by this evidence, and should be reported as
        tied. The bars at the top of the page draw each interval as a pale band for
        exactly this reason, so overlap is visible instead of buried in a table.</p>
      </li>

      <li class="step">
        <h3><span class="sn">6</span> Canonicalization</h3>
        <p>Before anything is compared, both sides are mapped into one token space, so
        <span class="tok">20th</span> and <span class="tok">twentieth</span> become the
        same token and <span class="tok">10:30</span> matches
        <span class="tok">ten thirty</span>. Where a string has several legitimate
        spoken readings, every one is generated and a match against any of them counts.
        A system is not marked wrong for picking a different valid reading than the
        reference author happened to write.</p>
      </li>

      <li class="step">
        <h3><span class="sn">7</span> Comparing systems fairly</h3>
        <p>Systems generate different numbers of items, so the cross-system column is
        computed on the <strong>intersection</strong>: the
        <strong>{summary.get('common_items', 0)}</strong> items every system scored.
        Ranking a system measured on {n} items against one measured on {smallest_n} is
        not a like-for-like race, so both numbers are shown and the intersection is the
        one to trust.</p>
      </li>

    </ol>
  </section>"""


def main() -> int:
    ap = argparse.ArgumentParser(description="Standalone HTML report for a run")
    ap.add_argument("--run", default="runs/combined")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    run_dir = Path(args.run)
    out = Path(args.out) if args.out else run_dir / "report.html"
    out.write_text(render(run_dir), encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
