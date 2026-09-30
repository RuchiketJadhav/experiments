"""Mock adapters, so the pipeline is testable with no API key and no model.

MockTTS writes a real (tone) wav so the audio layer, cache and screening are
genuinely exercised. MockASR returns a scripted transcript per item, which lets
a test assert on correct renderings, wrong renderings, silence and
hallucinations without needing speech.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from audio import write_wav


class MockTTS:
    """Writes a deterministic tone whose length tracks the text length."""

    normalizes_output = False

    def __init__(self, id: str = "mock-tts", revision: str = "v0",
                 sample_rate: int = 16_000, silent: bool = False,
                 seconds_per_char: float = 0.02) -> None:
        self.id = id
        self.revision = revision
        self.sample_rate = sample_rate
        self.silent = silent
        self.seconds_per_char = seconds_per_char
        self.calls = 0

    def synthesize(self, text: str, out_path: Path, sample_index: int = 0,
                   item_id: str | None = None) -> None:
        self.calls += 1
        n = max(int(self.sample_rate * self.seconds_per_char * max(len(text), 1)), 400)
        if self.silent:
            samples = np.zeros(n, dtype=np.float32)
        else:
            t = np.arange(n, dtype=np.float32) / self.sample_rate
            freq = 180.0 + 20.0 * sample_index
            samples = (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        write_wav(out_path, samples, self.sample_rate)


class MockASR:
    """Returns a scripted transcript per item id."""

    def __init__(self, transcripts: dict[str, str], id: str = "mock-asr",
                 revision: str = "v0", normalizes_output: bool = False,
                 default: str = "") -> None:
        self.id = id
        self.revision = revision
        self.normalizes_output = normalizes_output
        self._transcripts = transcripts
        self._default = default
        self.calls = 0

    def transcribe(self, wav_path: Path) -> str:
        self.calls += 1
        # The runner encodes the item id into the cached wav's stem prefix.
        stem = Path(wav_path).stem
        for item_id, text in self._transcripts.items():
            if stem.startswith(item_id):
                return text
        return self._default
