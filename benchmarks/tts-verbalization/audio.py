"""Audio IO for the bench.

Deliberately dependency-light: soundfile + numpy only. There is no ffmpeg,
librosa, scipy or torchaudio on this machine, so resampling is linear
interpolation done here.

Linear interpolation is adequate for downsampling speech to 16 kHz for ASR but
it is not a high-quality resampler. Prefer asking the TTS provider for 16 kHz
directly -- then no resampling happens at all and this code path is skipped.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf

ASR_SAMPLE_RATE = 16_000
TELEPHONY_SAMPLE_RATE = 8_000


def read_wav(path: Path | str) -> tuple[np.ndarray, int]:
    """Return (mono float32 in [-1,1], sample_rate)."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data.astype(np.float32), sr


def write_wav(path: Path | str, samples: np.ndarray, sr: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples.astype(np.float32), sr, subtype="PCM_16")


def resample(samples: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr or samples.size == 0:
        return samples
    n_out = int(round(samples.size * dst_sr / src_sr))
    if n_out <= 1:
        return samples[:1]
    src_idx = np.linspace(0.0, samples.size - 1, num=n_out, dtype=np.float64)
    return np.interp(src_idx, np.arange(samples.size), samples).astype(np.float32)


def load_for_asr(path: Path | str) -> np.ndarray:
    samples, sr = read_wav(path)
    return resample(samples, sr, ASR_SAMPLE_RATE)


def duration_s(path: Path | str) -> float:
    info = sf.info(str(path))
    return float(info.frames) / float(info.samplerate)


def rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


def content_hash(*parts: str) -> str:
    """Stable cache key. Any change to a part yields a different wav path, so a
    config edit can never silently reuse audio synthesized under old settings."""
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()[:20]


def file_sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
