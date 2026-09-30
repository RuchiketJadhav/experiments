"""wav2vec2 / CTC recognizer -- the PRIMARY evaluator.

Why primary rather than a robustness backup: a CTC acoustic model emits what
was said, as words, with no punctuation and no digit conversion. It writes
"MAY TWENTIETH TWENTY TWENTY THREE". Whisper writes "May 20th, 2023" -- it
re-normalizes speech back into written form, which is precisely the
transformation this benchmark is trying to measure. Scoring a correct TTS
against a spoken reference through a normalizing ASR penalizes it on every
item.

So for a verbalization benchmark a non-normalizing recognizer is a correctness
requirement, not a nicety.

Model weights download on first use (~360 MB for base-960h). CPU is fine for
this corpus: 540 short clips.
"""

from __future__ import annotations

import functools
from pathlib import Path

from audio import ASR_SAMPLE_RATE, load_for_asr

DEFAULT_MODEL = "facebook/wav2vec2-base-960h"


class Wav2Vec2ASR:
    normalizes_output = False

    def __init__(self, model: str = DEFAULT_MODEL, device: str = "cpu") -> None:
        self.id = f"wav2vec2:{model}"
        self.revision = model
        self._model_name = model
        self._device = device

    @functools.cached_property
    def _pipe(self):
        # Imported lazily so the rest of the pipeline (and its tests) run
        # without torch/transformers being touched.
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        processor = Wav2Vec2Processor.from_pretrained(self._model_name)
        model = Wav2Vec2ForCTC.from_pretrained(self._model_name).to(self._device)
        model.eval()
        return torch, processor, model

    def transcribe(self, wav_path: Path) -> str:
        torch, processor, model = self._pipe
        samples = load_for_asr(wav_path)
        inputs = processor(
            samples, sampling_rate=ASR_SAMPLE_RATE, return_tensors="pt", padding=True
        )
        with torch.no_grad():
            logits = model(inputs.input_values.to(self._device)).logits
        ids = torch.argmax(logits, dim=-1)          # greedy: deterministic
        return processor.batch_decode(ids)[0].strip()
