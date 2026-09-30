"""tts-bench UI -- run page.

    python -m streamlit run app.py

A front end over the existing engine, not a second implementation.

TWO THINGS THAT MATTER HERE:

1. Streamlit re-runs this whole script on EVERY widget interaction. A 171 MB
   zip must therefore be extracted once and cached, or uploading one turns into
   a re-extraction on every click and the app looks broken.
2. A key pasted into the config box lives in session state for this browser
   session and nowhere else: never written to disk, never echoed back, and it
   cannot reach runs/*/manifest.json because that manifest is a field allowlist.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import time
import traceback
from pathlib import Path

import streamlit as st
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import build_asr, build_items, build_tts, load_config  # noqa: E402
from ingest import ingest_zip  # noqa: E402
from run import RunPaths, run, write_manifest  # noqa: E402
from sampling import sample_items  # noqa: E402
from score import summarize  # noqa: E402

st.set_page_config(page_title="tts-bench", page_icon="🔊", layout="wide")

DEFAULT_ASR = [{"type": "wav2vec2", "model": "facebook/wav2vec2-base-960h",
                "device": "cpu"}]

SAMPLE_YAML = """\
tts:
  - id: my-tts
    type: http
    revision: my-model-v1
    endpoint: "https://api.example.com/v1/tts"
    headers:
      Authorization: "Bearer YOUR_KEY"
      Content-Type: application/json
    body:
      text: "{{text}}"
      model_id: my-model-v1
    response: audio
"""


@st.cache_resource(show_spinner="Loading the recognizer (once per session)…")
def get_asr() -> list:
    """Returns a LIST of adapters -- build_asr always does. Do not wrap it
    again at the call site; run() iterates it and would hit .id on a list."""
    return build_asr({"asr": DEFAULT_ASR})


@st.cache_data(show_spinner=False)
def get_items():
    return build_items({"dataset": {"type": "polynorm"}})


@st.cache_resource(show_spinner="Extracting and indexing the archive…")
def cached_ingest(digest: str, raw: bytes, sys_id: str):
    """Cached on the file's content hash, so a 171 MB zip is extracted ONCE
    rather than on every Streamlit rerun."""
    work = Path(tempfile.mkdtemp(prefix="ttsbench-ui-"))
    zpath = work / "upload.zip"
    zpath.write_bytes(raw)
    return ingest_zip(zpath, get_items(), work / "extracted", id=sys_id)


def fail(msg: str, exc: Exception | None = None) -> None:
    st.error(msg)
    if exc is not None:
        with st.expander("Full error detail"):
            st.code("".join(traceback.format_exception(exc)), language="text")


items_all = get_items()
n_cats = len({i.category for i in items_all})

# ============================================================ sidebar
with st.sidebar:
    st.header("Set up a run")

    mode = st.radio("Where does the audio come from?",
                    ["Saved config", "Upload a zip", "Call an API"])

    tts_adapters: list = []
    note = None

    if mode == "Saved config":
        cfgs = sorted(Path("configs").glob("*.yaml"))
        if not cfgs:
            st.info("Nothing in configs/.")
        else:
            pick = st.selectbox("File", [p.name for p in cfgs])
            try:
                tts_adapters = build_tts(load_config(Path("configs") / pick), items_all)
                note = f"{len(tts_adapters)} system(s) ready"
            except Exception as exc:                    # noqa: BLE001
                fail(f"Could not build from {pick}", exc)

    elif mode == "Upload a zip":
        up = st.file_uploader("Zip of audio", type=["zip"],
                              help="Either a manifest.csv, or <category>/<index>.wav folders")
        sys_id = st.text_input("System name", value="uploaded-system")
        if up is not None:
            raw = up.getvalue()
            digest = hashlib.sha256(raw).hexdigest()[:16]
            try:
                ing = cached_ingest(digest, raw, sys_id or "uploaded")
                tts_adapters = [ing.adapter]
                note = (f"{ing.audio_files} files · layout **{ing.layout}** · "
                        f"{ing.matched_items} usable · "
                        f"{len(ing.adapter.categories_covered())}/{n_cats} categories")
            except Exception as exc:                    # noqa: BLE001
                fail(f"Could not read that archive: {exc}", exc)

    else:
        text = st.text_area("Config (YAML)", value=SAMPLE_YAML, height=260)
        st.caption("Key stays in this session — never written to disk or into results.")
        if text.strip():
            try:
                cfg = yaml.safe_load(text) or {}
                if "tts" not in cfg:
                    raise ValueError("config needs a top-level `tts:` list")
                tts_adapters = build_tts(cfg, items_all)
                note = f"{len(tts_adapters)} system(s) ready"
            except Exception as exc:                    # noqa: BLE001
                fail(f"Config problem: {exc}", exc)

    if note:
        st.success(note)

    st.divider()
    full = st.toggle(f"Score all {len(items_all)} items", value=False)
    if full:
        st.warning("15–30 min per system on CPU. Cached after the first pass.")
    else:
        st.caption(f"20 items sampled across all {n_cats} categories.")
    items = items_all if full else sample_items(items_all, 20, seed=0)

    go = st.button("Run benchmark", type="primary", use_container_width=True,
                   disabled=not tts_adapters)

# ============================================================ header
st.title("TTS verbalization bench")
st.caption(
    "Does a system **say** dates, money, phone numbers and addresses correctly? "
    "Verbalization only — not audio quality or naturalness. "
    "See **Methodology** in the sidebar for how each number is derived."
)

# ============================================================ run
if go:
    asr_adapters = get_asr()
    out = Path("runs") / f"ui-{int(time.time())}"
    paths = RunPaths(out)
    bar = st.progress(0.0, text="starting…")
    try:
        counts = run(items, tts_adapters, asr_adapters, paths, progress=False,
                     on_progress=lambda d, t: bar.progress(
                         min(d / max(t, 1), 1.0), text=f"{d} / {t} records"))
        write_manifest(paths, items, tts_adapters, asr_adapters, 1, None, counts)
        st.session_state["last_run"] = str(out)
        st.session_state["last_counts"] = counts
    except Exception as exc:                            # noqa: BLE001
        bar.empty()
        fail(f"Run failed: {exc}", exc)
        st.stop()
    bar.empty()

# ============================================================ results
last = st.session_state.get("last_run")
if not last or not (Path(last) / "results.jsonl").exists():
    st.info("Pick a source in the sidebar, then **Run benchmark**. "
            "The saved config `local_audio.yaml` is the quickest look — its audio "
            "is already cached.")
    st.stop()

recs = [json.loads(l) for l in
        (Path(last) / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()]
summary = summarize(recs)
counts = st.session_state.get("last_counts", {})
scored = sum(1 for r in recs if r.get("status") == "scored")
excluded = len(recs) - scored

c1, c2, c3, c4 = st.columns(4)
c1.metric("Systems", len(summary["systems"]))
c2.metric("Scored", scored)
c3.metric("Excluded", excluded)
c4.metric("Elapsed", f"{counts.get('elapsed_s', 0):.0f}s")

st.subheader("Results")
rows = [{
    "System": s,
    "Correct": d["match_rate"],
    "95% CI": f"{d['ci95'][0]:.0%} – {d['ci95'][1]:.0%}",
    "Scored": d["scored"],
    "Strict sentence": d.get("sentence_match_rate"),
    "Excluded": sum(d["excluded"].values()),
} for s, d in sorted(summary["systems"].items(),
                     key=lambda kv: -kv[1]["match_rate"])]
st.dataframe(
    rows, use_container_width=True, hide_index=True,
    column_config={
        "Correct": st.column_config.ProgressColumn(
            "Correct", min_value=0.0, max_value=1.0, format="%.0f%%",
            help="Required verbalization present in the transcript"),
        "Strict sentence": st.column_config.NumberColumn(
            format="%.0f%%", help="Whole-sentence match; lower because recognizer "
                                  "errors anywhere in the carrier fail it"),
    })

if excluded:
    st.caption(
        f"Excluded, **not** counted as failures: `{summary['excluded_reasons']}`. "
        "Items the provider never generated, silent audio, suspected hallucinated "
        "transcripts, and items with nothing to normalize."
    )

cats = summary["categories"]
if len(cats) > 1:
    st.subheader("By category")
    sysnames = sorted(summary["systems"])
    st.dataframe(
        [{"Category": c, **{s: (d[s]["match_rate"] if d.get(s) else None)
                            for s in sysnames}}
         for c, d in sorted(cats.items())],
        use_container_width=True, hide_index=True,
        column_config={s: st.column_config.ProgressColumn(
            s, min_value=0.0, max_value=1.0, format="%.0f%%") for s in sysnames})
    st.caption("20 items per category at most — treat single categories as "
               "exploratory and pool before making a claim.")

st.subheader("Every item")
st.caption("The drill-down is what makes a number believable.")
left, right = st.columns([1, 3])
only_miss = left.toggle("Misses only", value=False)
detail = [{
    "": "❌" if r.get("target_match") is False else
        ("✅" if r.get("target_match") else "—"),
    "Category": r["category"],
    "Written": r["written"],
    "Must say": r.get("target", ""),
    "Heard": r.get("hypothesis") or "",
    "Status": r.get("status"),
    "Reason": r.get("reason", ""),
} for r in recs if not (only_miss and r.get("target_match"))]
st.dataframe(detail, use_container_width=True, hide_index=True,
             column_config={"": st.column_config.TextColumn(width="small")})
st.caption(f"Run directory: `{last}`")
