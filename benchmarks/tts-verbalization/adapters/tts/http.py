"""Generic HTTP TTS adapter -- a new provider costs a config block, not code.

Config shape:
    id: cartesia-sonic-3.6
    revision: sonic-3.6
    endpoint: https://api.example.com/v1/tts
    method: POST
    headers: {Authorization: "Bearer ${MY_KEY}"}   # ${VAR} reads the env
    body: {text: "{{text}}", model: "sonic-3.6", sample_rate: 16000}
    response: audio            # raw audio bytes in the body
    # or: response: {json_path: "audio.data", encoding: base64}

SECURITY NOTE: this takes an arbitrary URL. That is harmless for a CLI run on
your own machine. If this is ever exposed as a hosted service it becomes an
SSRF primitive and needs a scheme/host allowlist plus blocking of link-local
(169.254.169.254) and private ranges. Do not host this as written.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

import requests

from adapters.httputil import dig as _dig
from adapters.httputil import expand_env as _expand_env



def _fill_template(value: Any, text: str) -> Any:
    if isinstance(value, str):
        return value.replace("{{text}}", text)
    if isinstance(value, dict):
        return {k: _fill_template(v, text) for k, v in value.items()}
    if isinstance(value, list):
        return [_fill_template(v, text) for v in value]
    return value





def write_audio_atomic(out_path, data: bytes) -> None:
    """Write the clip to a temporary name, then rename it into place.

    A plain write_bytes() leaves a truncated file behind when the process dies
    mid-write, and the watchdog kills the runner by design. The next pass sees
    out_path.exists(), skips synthesis, and the screener marks the stub
    screened_audio -- which is NOT in run.py's TRANSIENT set, so --retry-errors
    never revisits it. One interrupted write became a permanent hole in the
    data, scored as the model's fault. os.replace() is atomic on POSIX and on
    Windows, so the file either is not there or is complete.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    tmp.write_bytes(data)
    os.replace(tmp, out_path)

class HttpTTS:
    def __init__(self, cfg: dict) -> None:
        self.id = cfg["id"]
        self.revision = str(cfg.get("revision", "unknown"))
        self._endpoint = cfg["endpoint"]
        self._method = cfg.get("method", "POST").upper()
        self._headers = _expand_env(cfg.get("headers", {}))
        self._body = cfg.get("body", {})
        self._response = cfg.get("response", "audio")
        self._timeout = float(cfg.get("timeout_s", 60))

    def synthesize(self, text: str, out_path: Path, sample_index: int = 0,
                   item_id: str | None = None) -> None:
        body = _fill_template(self._body, text)
        resp = requests.request(
            self._method, self._endpoint, headers=self._headers,
            json=body, timeout=self._timeout,
        )
        resp.raise_for_status()

        if self._response == "audio":
            data = resp.content
        else:
            spec = self._response
            payload = _dig(json.loads(resp.text), spec["json_path"])
            data = (base64.b64decode(payload)
                    if spec.get("encoding") == "base64" else payload.encode())

        if not data:
            raise ValueError(f"{self.id} returned an empty body for {item_id!r}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_audio_atomic(out_path, data)
