"""Adapter protocols.

Shape follows evals/stt/providers/base.py in the dograh repo (STTProvider /
TranscriptionResult) so the two stay recognisable to the same reader.

Every adapter carries an `id` and a `revision`. Both land in the run manifest:
comparing a number against a past run is only meaningful if you can see which
model produced it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


class NotGenerated(Exception):
    """The provider never produced audio for this item.

    Distinct from a synthesis error on purpose. A row the upstream harness
    failed to generate (rate limit, SSL drop, gRPC timeout) is not evidence
    about the TTS model, so it is recorded as excluded rather than scored as a
    miss. Folding these into the rate would punish a model for the network.
    """


@runtime_checkable
class TTSAdapter(Protocol):
    id: str
    revision: str

    def synthesize(self, text: str, out_path: Path, sample_index: int = 0) -> None:
        """Write a wav for `text` to `out_path`.

        `sample_index` distinguishes repeat draws. Hosted APIs expose no seed,
        so repeats are SAMPLES, not seeds -- they measure variance, they do not
        make a run reproducible. Local models may use it as an actual seed.
        """


@runtime_checkable
class ASRAdapter(Protocol):
    id: str
    revision: str
    normalizes_output: bool   # True if it rewrites speech into written form

    def transcribe(self, wav_path: Path) -> str:
        ...
