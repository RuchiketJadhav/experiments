"""Methodology, guardrails, and what each number cannot tell you."""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

st.set_page_config(page_title="Methodology · tts-bench", page_icon="📐",
                   layout="wide")

st.title("Methodology")
st.caption("Every rule here exists because leaving it out produced a wrong number.")

# ----------------------------------------------------------------- scope
st.header("What is measured")
st.markdown(
"""
**Verbalization correctness.** Given text containing a non-standard token — a date,
a price, a phone number, a URL — did the system *say* it correctly?

**What is not measured:** audio quality, naturalness, prosody, speaker similarity,
latency. A system can score perfectly here and sound robotic.

Two reasons to keep that boundary explicit. SP-MCQA (arXiv:2510.26190) found that a
low word error rate does not guarantee a listener recovers the key information. And
arXiv:2509.18531 found that optimizing a TTS system against WER-family rewards
*measurably collapses prosody into monotone speech*. This number is a guardrail to
check, never a target to optimize.
"""
)

# ----------------------------------------------------------------- pipeline
st.header("The pipeline")
st.code(
"""  dataset ──▶ TTS adapter ──▶ audio (cached by content hash)
     │                              │
     │                              ▼
     │                      screening pass ──▶ excluded (reported separately)
     │                              │
     │                              ▼
     │                       wav2vec2 (CTC)
     │                              │
     └── written + spoken ──▶ target extraction ──▶ per-item match ──▶ rates + CIs""",
    language="text")

# ----------------------------------------------------------------- ASR
st.header("Why a CTC recognizer, not Whisper")
st.markdown(
"""
The reference is a *spoken* form. A normalizing recognizer converts speech back into
*written* form, which is exactly the transformation under test:
"""
)
st.code(
"""  reference : The event is on May twentieth twenty twenty three.
  Whisper   : The event is on May 20th, 2023.        <- re-normalized
  wav2vec2  : THE EVENT IS ON MAY TWENTIETH TWENTY TWENTY THREE""",
    language="text")
st.markdown(
"""
Scored word-by-word through Whisper, a **perfectly correct** system is penalized on
every item. So a non-normalizing CTC model is a *correctness requirement* here, not a
robustness nicety — which inverts the usual advice to prefer Whisper.

This matters for a second reason. arXiv:2607.08256 showed that on identical audio,
F5-TTS scores **2.06% WER under Whisper-large-v3, 1.52% under wav2vec2-lv60 and 1.92%
under HuBERT-large**, and that the *preferred system reverses* depending on which
family judges. Representation similarity does not predict agreement, so "pick a
similar recognizer" is not a defence. Their recommendation is at least two families
with disjoint lineages.

**Current limitation:** only one lineage is wired. Cross-lineage disagreement is
therefore not measurable yet, and a single-evaluator ranking is weaker evidence than
a two-lineage one.
"""
)

# ----------------------------------------------------------------- scoring
st.header("How a single item is scored")
st.markdown(
"""
**Step 1 — find the target by surface diff.** Compare the written and spoken forms as
raw words. What the spoken form adds is what the system had to get right; everything
else is carrier.
"""
)
st.code(
"""  written : Let's meet at 10:30 tomorrow.        -> target: "ten thirty"
  written : The price is $10.99.                 -> target: "ten dollars and ninety nine cents"
  written : The deadline is 2024-12-31.          -> target: "December thirty first twenty twenty four" """,
    language="text")
st.markdown(
"""
**Step 2 — score the target canonically.** Number words and digits are mapped into one
space, so "20th" and "twentieth" agree, and word order inside the target does not
matter ("July third" and "third July" are both correct).

**Why not score the whole sentence?** Because carrier noise from the recognizer fails
correct systems. This is a real case from a real run:
"""
)
st.code(
"""  reference : The deadline is December thirty first twenty twenty four.
  heard     : THE DEAD LINE IS DECEMBER THIRTY FIRST TWENTY TWENTY FOUR
  whole-sentence scoring : MISS   <- wav2vec2 split "deadline"
  target scoring         : MATCH  <- the date is correct, and that is what is tested""",
    language="text")
st.markdown(
    "The strict whole-sentence rate is still reported beside the headline, as the "
    "`strict sentence` column, so the gap between them is visible rather than hidden."
)

# ----------------------------------------------------------------- canon
st.header("The canonicalizer")
st.markdown(
"""
- **Number words → digits**, including compounds ("one hundred twenty three" → 123)
  and year pairs ("nineteen ninety five" → 1995).
- **Multiple valid readings.** Each numeric run expands to a *set* of acceptable
  forms, so "one two three" matches 123 or 1 2 3. This is how the
  one-reference-per-item limitation of the dataset is absorbed.
- **Symbols carry several readings** — `.` is dot, point, or silent; `:` is silent,
  colon, or "to" — rather than the code guessing which one a speaker used.
- **Ordinality is preserved** (`20` vs `20#o`), because Cardinal and Ordinal are
  distinct categories and collapsing them would hide real errors.
- **Digit merging is per-category, not global.** On for identifiers (ISBN, phone,
  serial) where grouping is inaudible. Off elsewhere on purpose: with it on,
  "two zero two three" would match "twenty twenty three", and reading a year
  digit-by-digit is precisely the failure this benchmark exists to catch.
"""
)

# ----------------------------------------------------------------- screening
st.header("Screening runs before scoring, never after")
st.markdown(
"""
If synthesis failed, the audio is silent, or the recognizer hallucinated fluent text
over near-silence, then edit distance is measuring a category error rather than
intelligibility (the INSV framework, arXiv:2605.26978, makes this the ordering rule).

Checks: file exists and is non-empty, duration is plausible for the text length, the
audio is not effectively silent, and the transcript is not more than 2× the reference
length (the signature of a hallucinated transcript).

**Excluded items are reported in their own column and never averaged in as zeros.**
An item the provider never generated — a rate limit, an SSL drop — is not evidence
about the model. Folding those in would punish a system for its harness's network.
"""
)

# ----------------------------------------------------------------- stats
st.header("Rates, denominators and comparability")
st.markdown(
"""
- **Denominator** is `scored / generated`, never `/ 540`. One system in the local set
  generated 514 of 540; dividing by 540 would report it as worse for reasons that have
  nothing to do with the model.
- **Bootstrap 95% CI**, 1000 resamples, on every rate. With 20 items per category the
  interval is what stops a two-item gap being read as a finding. If two intervals
  overlap, the systems are not distinguishable on this evidence.
- **Like-for-like column.** The cross-system comparison is also computed on the
  intersection of items every system generated, because comparing one system on 539
  items against another on 514 is not a fair race.
- **Repeats are samples, not seeds.** Most hosted APIs expose no seed, so repeated
  draws measure variance and do not make a run reproducible. (ElevenLabs is an
  exception: it accepts a best-effort `seed`.)
"""
)

# ----------------------------------------------------------------- limits
st.header("Limitations — read these before quoting a number")
st.warning(
"""
**1. Function words are currently required in the target.** The reference for
"15th August" is "the fifteenth **of** August". A system saying "fifteenth August" is
correct English but scores as a miss. On one 10-item run, **4 of 10 misses would flip**
if function words were dropped. This is an open decision, disclosed rather than
quietly tuned away.

**2. One recognizer lineage.** Cross-family disagreement — the thing arXiv:2607.08256
says you must check — cannot be measured yet.

**3. Eleven of 540 items have no target.** Their written and spoken forms are
identical, so they cannot distinguish a right rendering from a wrong one. They are
excluded and counted rather than guessed at.

**4. Resampling is linear interpolation.** This machine has no ffmpeg, librosa, scipy
or torchaudio. Request 16 kHz from the provider and the path is skipped entirely.

**5. Twenty items per category.** Per-category numbers are exploratory. Pool
categories before making any claim.

**6. The recognizer is not perfect.** It writes "DEAD LINE" for "deadline" and
"HEY MARRIED" for "They married". Target scoring removes most of this, but a
recognizer error *inside* the target span will still cost a system a point.
"""
)

# ----------------------------------------------------------------- data
st.header("Dataset and licence")
st.markdown(
"""
[PolyNorm-Bench](https://github.com/apple/ml-speech-polynorm-bench) (Apple): 540 items,
27 categories, 20 each, generated then verified by human language experts. Each row
pairs a **written** form with its correct **spoken** form, which is what makes
verbalization measurable at all.

**Licence: CC BY-NC-ND 4.0.** Private reformatting is permitted — the licence allows
producing adapted material but *not sharing* it. Results computed from it may be
shown; the dataset and anything derived from it must not be redistributed, and
commercial use is restricted.
"""
)

st.header("References")
st.markdown(
"""
- arXiv:2607.08256 — *Best-of-N TTS Evaluation is Confounded by ASR Family Alignment*
- arXiv:2605.26978 — *PashtoTTS-Bench* (the INSV screen-before-scoring framework)
- arXiv:2510.26190 — *SP-MCQA: Evaluating Intelligibility of TTS Beyond the Word Level*
- arXiv:2509.18531 — *No Verifiable Reward for Prosody*
- arXiv:2503.03250 — *Good practices for evaluation of synthesized speech*
"""
)
