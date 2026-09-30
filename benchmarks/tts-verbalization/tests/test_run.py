"""End-to-end pipeline test using mock adapters.

Covers the properties that actually protect a long single-operator run:
caching, resume, per-item error isolation, and screening BEFORE scoring.

Run: python tests/test_run.py
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from adapters.mock import MockASR, MockTTS  # noqa: E402
from run import RunPaths, run, write_manifest  # noqa: E402
from schema import Item  # noqa: E402

failures: list[str] = []
checks = 0


def check(cond: bool, label: str) -> None:
    global checks
    checks += 1
    if not cond:
        failures.append(label)


ITEMS = [
    Item("i1", "Date", "The event is on 05/20/2023.",
         "The event is on May twentieth twenty twenty three.", "test"),
    Item("i2", "Currency", "The price is $10.99.",
         "The price is ten dollars and ninety nine cents.", "test"),
    Item("i3", "Time", "The train departs at 15:45.",
         "The train departs at fifteen forty five.", "test"),
]

TRANSCRIPTS = {
    # correct, but written back in digit form by a normalizing ASR
    "i1": "The event is on May 20th, 2023.",
    # wrong: cents dropped
    "i2": "The price is $10.",
    # correct
    "i3": "The train departs at fifteen forty five.",
}


def records(paths: RunPaths) -> dict[str, dict]:
    return {json.loads(l)["item_id"]: json.loads(l)
            for l in paths.results.read_text(encoding="utf-8").splitlines() if l.strip()}


tmp = Path(tempfile.mkdtemp(prefix="ttsbench-"))
try:
    # ---------------------------------------------------------------- scoring
    paths = RunPaths(tmp / "r1")
    tts = MockTTS()
    asr = MockASR(TRANSCRIPTS, normalizes_output=True)
    counts = run(ITEMS, [tts], [asr], paths, progress=False)

    check(counts["written"] == 3, f"expected 3 records, got {counts['written']}")
    recs = records(paths)
    check(recs["i1"]["status"] == "scored", "i1 should be scored")
    check(recs["i1"]["match"] is True,
          f"i1 correct-but-digit-form should MATCH, got {recs['i1']}")
    check(recs["i2"]["match"] is False, "i2 (cents dropped) should NOT match")
    check(recs["i3"]["match"] is True, "i3 should match")
    check(recs["i1"]["wer"] == 0.0, "i1 wer should be 0.0")
    check(recs["i2"]["wer"] > 0.0, "i2 wer should be > 0")

    # ------------------------------------------------------------ cache/resume
    synth_calls_first = tts.calls
    asr_calls_first = asr.calls
    counts2 = run(ITEMS, [tts], [asr], paths, progress=False)
    check(tts.calls == synth_calls_first,
          f"re-run must synthesize nothing; {tts.calls - synth_calls_first} extra calls")
    check(asr.calls == asr_calls_first,
          f"re-run must not re-transcribe; {asr.calls - asr_calls_first} extra calls")
    check(counts2["written"] == 0, f"re-run wrote {counts2['written']} records")
    check(counts2["skipped_done"] == 3, "re-run should skip 3 done keys")

    # A new item is processed while the existing ones stay skipped.
    extra = ITEMS + [Item("i4", "Cardinal", "There are 42 files.",
                          "There are forty two files.", "test")]
    asr2 = MockASR({**TRANSCRIPTS, "i4": "There are 42 files."},
                   normalizes_output=True)
    counts3 = run(extra, [tts], [asr2], paths, progress=False)
    check(counts3["written"] == 1, f"expected 1 new record, got {counts3['written']}")
    check(records(paths)["i4"]["match"] is True, "i4 should match")

    # A truncated final line (process killed mid-write) must not break resume.
    with paths.results.open("a", encoding="utf-8") as fh:
        fh.write('{"key": "trunc')
    counts4 = run(extra, [tts], [asr2], paths, progress=False)
    check(counts4["written"] == 0, "truncated tail must not cause reprocessing")

    # ------------------------------------------------------- screening: silence
    p2 = RunPaths(tmp / "r2")
    silent = MockTTS(id="silent-tts", silent=True)
    counts5 = run(ITEMS, [silent], [MockASR(TRANSCRIPTS)], p2, progress=False)
    r2 = records(p2)
    check(all(r["status"] == "screened_audio" for r in r2.values()),
          f"silent audio must be screened, got {[r['status'] for r in r2.values()]}")
    check(all(r["match"] is None for r in r2.values()),
          "screened items must not carry a score")
    check(counts5["screened_audio"] == 3, "expected 3 audio screen-outs")

    # ------------------------------------------- screening: ASR hallucination
    p3 = RunPaths(tmp / "r3")
    babble = MockASR({"i1": "the " * 60}, default="ok")
    run([ITEMS[0]], [MockTTS(id="t3")], [babble], p3, progress=False)
    check(records(p3)["i1"]["status"] == "screened_transcript",
          "runaway transcript must be screened as a hallucination")

    # --------------------------------------------------- per-item isolation
    class BoomTTS(MockTTS):
        def synthesize(self, text, out_path, sample_index=0, item_id=None):
            if item_id == "i2":
                raise RuntimeError("429 rate limited")
            super().synthesize(text, out_path, sample_index, item_id)

    p4 = RunPaths(tmp / "r4")
    run(ITEMS, [BoomTTS(id="boom")], [MockASR(TRANSCRIPTS)], p4, progress=False)
    r4 = records(p4)
    check(len(r4) == 3, f"a failing item must not abort the run; got {len(r4)}")
    check(r4["i2"]["status"] == "synthesis_error", "i2 should record synthesis_error")
    check("429" in (r4["i2"]["reason"] or ""), "failure reason should be preserved")
    check(r4["i1"]["status"] == "scored" and r4["i3"]["status"] == "scored",
          "items either side of a failure must still be scored")

    # ------------------------------------------ nested-list adapter guard
    # The UI hit this: build_asr() returns a list and the call site wrapped it
    # again, producing "'list' object has no attribute 'id'" from deep in the
    # loop. The guard must name the fix instead.
    p5 = RunPaths(tmp / "r5")
    a_ok = MockASR(TRANSCRIPTS)
    try:
        run(ITEMS, [MockTTS(id="t5")], [[a_ok]], p5, progress=False)  # type: ignore[list-item]
        checks += 1
        failures.append("nested adapter list should raise TypeError")
    except TypeError as exc:
        checks += 1
        check("flat list" in str(exc), f"guard should explain the fix; got {exc}")
    check(not (p5.results).exists() or not p5.results.read_text(encoding="utf-8").strip(),
          "a rejected run must not write records")

    # ------------------------------------------------------------- manifest
    write_manifest(p4, ITEMS, [BoomTTS(id="boom")], [MockASR(TRANSCRIPTS)], 1,
                   None, {"written": 3})
    man = json.loads(p4.manifest.read_text(encoding="utf-8"))
    check(man["items"] == 3 and man["samples_per_item"] == 1, "manifest basics")
    check("seed" in man["note"], "manifest must record the samples-vs-seeds caveat")
    blob = json.dumps(man).lower()
    check("authorization" not in blob and "bearer" not in blob,
          "manifest must not carry credentials")

    print(f"{checks - len(failures)}/{checks} checks passed")
    if failures:
        print(f"\n{len(failures)} FAILED:\n")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("all passed")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# --------------------------------------------- transient failures are retryable
# A long unattended run WILL hit a provider blip. Recording that as a final
# outcome means a resume skips those items and the run finishes with silent
# holes, which is worse than failing loudly. Deterministic outcomes must stay
# done, or a resume would pay to reproduce the same verdict.
from run import TRANSIENT, load_done  # noqa: E402

p6 = RunPaths(tmp / "r6")
p6.ensure()
with p6.results.open("w", encoding="utf-8") as fh:
    for key, status in (("k-ok", "scored"), ("k-net", "asr_error"),
                        ("k-synth", "synthesis_error"), ("k-quiet", "screened_audio"),
                        ("k-missing", "not_generated"), ("k-notarget", "no_target")):
        fh.write(json.dumps({"key": key, "status": status}) + "\n")

plain = load_done(p6.results)
check(len(plain) == 6, f"default resume treats every record as done, got {len(plain)}")

retry = load_done(p6.results, retry_errors=True)
check("k-net" not in retry, "asr_error must be retried with --retry-errors")
check("k-synth" not in retry, "synthesis_error must be retried with --retry-errors")
check("k-ok" in retry, "a scored record must never be re-run")
for k in ("k-quiet", "k-missing", "k-notarget"):
    check(k in retry, f"{k} is a deterministic outcome and must stay done")
check(TRANSIENT == {"asr_error", "synthesis_error"},
      "only provider/network failures may be classed transient")

# a truncated tail must still not break the retry path
with p6.results.open("a", encoding="utf-8") as fh:
    fh.write('{"key": "trunc')
check(len(load_done(p6.results, retry_errors=True)) == 4,
      "a truncated line must be dropped, not crash the retry resume")

print(f"{checks - len(failures)}/{checks} checks passed (retry-errors)")
if failures:
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
