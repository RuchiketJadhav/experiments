"""Screening pass -- runs BEFORE scoring, never after.

Rationale (INSV, arXiv:2605.26978): if synthesis failed, produced near-silence,
or the recognizer hallucinated fluent text over nothing, then edit distance is
measuring a category error rather than intelligibility. A screened-out item is
reported as its own outcome and excluded from the score, never folded in as a
bad WER -- averaging failures into the metric is how a broken pipeline looks
like a merely mediocre TTS.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from audio import duration_s, load_for_asr, rms

MIN_DURATION_S = 0.20
MAX_DURATION_PER_CHAR_S = 0.60      # generous upper bound on speaking rate
MIN_RMS = 1e-4                       # below this the file is effectively silent
MAX_HYP_REF_TOKEN_RATIO = 2.0        # Whisper invents text over near-silence


@dataclass(frozen=True)
class ScreenResult:
    ok: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.ok


OK = ScreenResult(True)


def screen_audio(path: Path | str, written: str) -> ScreenResult:
    p = Path(path)
    if not p.exists():
        return ScreenResult(False, "synthesis_missing")
    if p.stat().st_size == 0:
        return ScreenResult(False, "empty_file")
    try:
        dur = duration_s(p)
    except Exception as exc:                     # unreadable/corrupt container
        return ScreenResult(False, f"unreadable_audio:{type(exc).__name__}")
    if dur < MIN_DURATION_S:
        return ScreenResult(False, f"too_short:{dur:.2f}s")
    if written and dur > MAX_DURATION_PER_CHAR_S * max(len(written), 1):
        return ScreenResult(False, f"implausibly_long:{dur:.2f}s")
    try:
        if rms(load_for_asr(p)) < MIN_RMS:
            return ScreenResult(False, "silent")
    except Exception as exc:
        return ScreenResult(False, f"unreadable_audio:{type(exc).__name__}")
    return OK


def screen_transcript(hypothesis: str, reference: str) -> ScreenResult:
    if not hypothesis.strip():
        return ScreenResult(False, "empty_transcript")
    n_hyp, n_ref = len(hypothesis.split()), max(len(reference.split()), 1)
    if n_hyp > MAX_HYP_REF_TOKEN_RATIO * n_ref:
        return ScreenResult(False, f"hallucination_suspect:{n_hyp}v{n_ref}")
    return OK
