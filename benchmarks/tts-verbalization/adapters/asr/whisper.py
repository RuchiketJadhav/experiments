"""Whisper -- the SECOND recognizer lineage.

WHY THIS EXISTS, AND WHY THE ORIGINAL OBJECTION NO LONGER HOLDS.

Whisper was rejected early in this project for a good reason: it is a seq2seq
model with a language model attached, so it rewrites speech back into WRITTEN
form. Say "May twentieth twenty twenty three" and it returns "May 20th, 2023".
Scoring against a spoken reference through a recognizer that undoes the exact
transformation under test penalizes a correct system on every item.

That objection is now stale, for two reasons:

1. canonicalize.py did not exist when the choice was made. It maps written and
   spoken forms into one token space, so "20th" and "twentieth" are already the
   same token. Whisper's normalization is no longer damage; it is a form the
   scorer understands.
2. A listening audit of 40 clips found that ~64% of the failures the CTC model
   reported were speech the systems had said CORRECTLY. wav2vec2-base cannot
   hear spelled letters ("I S B N" -> "AIASPAN"), cannot separate homophones
   with no language model ("two" -> "TO"), and garbles ordinary words
   ("ninth" -> "NIGHT DENING"). The benchmark was measuring the recognizer.

The language model that made Whisper unusable under whole-string scoring is
precisely what fixes all three failure modes. Kept as a SECOND lineage rather
than a replacement: arXiv:2607.08256 shows TTS rankings reverse across ASR
families, so the point is to compare the two and report where they disagree,
not to swap one single point of failure for another.

Weights download on first use. small.en is ~500 MB, medium.en ~1.5 GB.
"""

from __future__ import annotations

import functools
from pathlib import Path

from audio import ASR_SAMPLE_RATE, load_for_asr

DEFAULT_MODEL = "openai/whisper-small.en"


class WhisperASR:
    # TRUE, and it matters: run.py and the report both surface this so a reader
    # knows the transcript has been rewritten into written form before scoring.
    normalizes_output = True

    def __init__(self, model: str = DEFAULT_MODEL, device: str = "cpu") -> None:
        self.id = f"whisper:{model}"
        self.revision = model
        self._model_name = model
        self._device = device

    @functools.cached_property
    def _pipe(self):
        # Lazy, so the rest of the pipeline and its tests never touch torch.
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        processor = WhisperProcessor.from_pretrained(self._model_name)
        model = WhisperForConditionalGeneration.from_pretrained(
            self._model_name).to(self._device)
        model.eval()
        return torch, processor, model

    def transcribe(self, wav_path: Path) -> str:
        torch, processor, model = self._pipe
        samples = load_for_asr(wav_path)
        inputs = processor(samples, sampling_rate=ASR_SAMPLE_RATE,
                           return_tensors="pt")
        feats = inputs.input_features.to(self._device)
        kw = {}
        # The .en checkpoints are English-only and reject language/task kwargs;
        # the multilingual ones need them pinned or they will occasionally
        # decide the clip is another language and translate it.
        if not self._model_name.endswith(".en"):
            kw = {"language": "en", "task": "transcribe"}
        with torch.no_grad():
            ids = model.generate(
                feats,
                # Greedy and bounded. Sampling would make the benchmark
                # irreproducible; the cap stops the known failure where Whisper
                # loops on near-silence and emits the same phrase forever.
                do_sample=False,
                num_beams=1,
                max_new_tokens=128,
                repetition_penalty=1.1,
                **kw,
            )
        return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
