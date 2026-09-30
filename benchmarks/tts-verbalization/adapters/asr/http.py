"""Generic HTTP ASR adapter -- a new recognizer costs a config block, not code.

Exists because the local models could not both fit: wav2vec2 plus
whisper-small.en is ~1.4 GB resident and the OS killed the run at 39% on a
5.8 GB machine. Two hosted lineages cost about $1.20 for this corpus and are
not bounded by local RAM at all.

TWO UPLOAD SHAPES, because providers disagree:

    upload: multipart   OpenAI -- the wav as a `file` part plus form fields
    upload: raw         Deepgram -- the wav bytes as the request body

THE FIELD THAT MATTERS MOST IS `normalizes_output`.

A recognizer that applies inverse text normalization rewrites "five twenty
twenty twenty three" into "5/20/2023" -- which is to say it can REPAIR a
verbalization error and make a broken system look correct. That is a false
positive, the costliest error this benchmark can make. Where a provider offers
switches (Deepgram: smart_format, numerals, punctuate) turn them OFF in the
endpoint query string; where it does not (OpenAI whisper-1), declare
`normalizes_output: true` and validate against human-labelled clips before
trusting a number.

SECURITY: takes an arbitrary URL from config. See adapters/httputil.py.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

from adapters.httputil import dig, expand_env, request_with_retry



def _as_pcm16(wav_path: Path) -> bytes:
    """The clip's bytes, re-encoded to 16-bit PCM if it is not already.

    chatterbox-multilingual returns 32-bit FLOAT wav; every other system
    in the set returns PCM_16. libsndfile reads float wav happily, so
    screen.py passes it, and then OpenAI's transcription endpoint
    rejected 293 of those clips with a bare 400 while accepting 369
    identical-looking ones. Nothing about the text, the size or the file's
    integrity predicted which -- the only thing that separated that model
    from the six with zero 400s was the sample format.

    Converting also removes a quieter problem: the two recognizers were
    being handed different encodings of the same speech, which is a
    confound in the very comparison they exist to make. Both now receive
    identical bytes.
    """
    raw = wav_path.read_bytes()
    try:
        import soundfile as sf
        if sf.info(str(wav_path)).subtype == "PCM_16":
            return raw
        audio, rate = sf.read(str(wav_path), dtype="float32")
        buf = io.BytesIO()
        sf.write(buf, audio, rate, format="WAV", subtype="PCM_16")
        return buf.getvalue()
    except Exception:
        # Unreadable here is not our verdict to make: send the original
        # and let the provider's error say what is wrong with it.
        return raw

class HttpASR:
    def __init__(self, cfg: dict) -> None:
        self.id = cfg["id"]
        self.revision = str(cfg.get("revision", cfg.get("id", "unknown")))
        # No default. A missing declaration would silently claim the recognizer
        # does not rewrite speech, which is the assumption that makes an ITN
        # false positive invisible.
        if "normalizes_output" not in cfg:
            raise ValueError(
                f"asr {self.id!r} must declare normalizes_output: whether it "
                f"rewrites speech into written form decides how its output can "
                f"be scored")
        self.normalizes_output = bool(cfg["normalizes_output"])

        self._endpoint = cfg["endpoint"]
        self._method = cfg.get("method", "POST").upper()
        self._headers = expand_env(cfg.get("headers", {}))
        self._upload = cfg.get("upload", "multipart")
        if self._upload not in ("multipart", "raw"):
            raise ValueError(f"asr {self.id!r}: upload must be "
                             f"'multipart' or 'raw', got {self._upload!r}")
        self._form = expand_env(cfg.get("form", {}))
        self._file_field = cfg.get("file_field", "file")
        self._json_path = cfg.get("json_path")
        self._timeout = float(cfg.get("timeout_s", 120))
        self._attempts = int(cfg.get("attempts", 5))

    def transcribe(self, wav_path: Path) -> str:
        data = _as_pcm16(Path(wav_path))
        kwargs: dict = {"headers": self._headers}
        if self._upload == "multipart":
            kwargs["files"] = {self._file_field: (Path(wav_path).name, data,
                                                  "audio/wav")}
            if self._form:
                kwargs["data"] = self._form
        else:
            kwargs["data"] = data

        resp = request_with_retry(self._method, self._endpoint,
                                  attempts=self._attempts,
                                  timeout=self._timeout, **kwargs)

        if not self._json_path:
            return resp.text.strip()
        try:
            payload = dig(json.loads(resp.text), self._json_path)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            # Name the path and show a bounded slice of the body. Without this
            # a provider changing its response shape looks like every item
            # failing to transcribe.
            raise ValueError(
                f"{self.id}: could not read {self._json_path!r} from the "
                f"response ({type(exc).__name__}); body began "
                f"{resp.text[:200]!r}") from exc
        if not isinstance(payload, str):
            raise ValueError(f"{self.id}: {self._json_path!r} is "
                             f"{type(payload).__name__}, expected text")
        return payload.strip()
