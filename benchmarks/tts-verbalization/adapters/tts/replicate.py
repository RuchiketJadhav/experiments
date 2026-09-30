"""Replicate TTS adapter.

The generic HTTP adapter cannot do this. Replicate is ASYNCHRONOUS: you POST a
prediction, it returns an id, you poll until status is "succeeded", and the
result is a URL you then have to fetch. Even with `Prefer: wait` (which blocks
up to ~60s and usually returns a finished prediction) the output is still a
URL, so a second request is always needed.

BILLING, because it decides which wrapper to pick:

  Official models (qwen/, resemble-ai/chatterbox*) bill per unit -- qwen3-tts
  is $0.02 per thousand INPUT characters -- and are always on, so no boot cost.
  Community models bill by hardware-second INCLUDING boot. The same model
  packaged twice can differ 15x: alphanumericuser/kokoro-82m advertises
  $0.025/run while kjjk10/kokoro-82m runs on a T4 at $0.000225/s.

So this adapter records `metrics.predict_time` from every prediction. Measured
seconds are the only honest basis for estimating what a full run will cost.

Config:
    id: kokoro-82m
    type: replicate
    model: jaaari/kokoro-82m        # owner/name -> latest version
    version: 19a5a56d...            # optional, pins an exact version
    revision: kokoro-v1.0/af_heart  # goes in the cache key, so changing the
                                    # voice does not silently reuse old audio
    input:
      text: "{{text}}"
      voice: af_heart
    output_path: ""                 # "" if output is a URL string, "0" if a list
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import requests

from adapters.httputil import dig, expand_env, request_with_retry

API = "https://api.replicate.com/v1"


def _fill(value: Any, text: str) -> Any:
    if isinstance(value, str):
        return value.replace("{{text}}", text)
    if isinstance(value, dict):
        return {k: _fill(v, text) for k, v in value.items()}
    if isinstance(value, list):
        return [_fill(v, text) for v in value]
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

class ReplicateTTS:
    def __init__(self, cfg: dict) -> None:
        self.id = cfg["id"]
        self.model = cfg.get("model", "")
        self.version = cfg.get("version")
        if not self.model and not self.version:
            raise ValueError(f"replicate tts {self.id!r} needs `model` or `version`")
        self.revision = str(cfg.get("revision", self.version or self.model))
        self._input = cfg.get("input", {"text": "{{text}}"})
        self._output_path = str(cfg.get("output_path", ""))
        self._timeout = float(cfg.get("timeout_s", 300))
        self._poll_every = float(cfg.get("poll_s", 1.0))
        token = expand_env(cfg.get("token", "${REPLICATE_API_TOKEN}"))
        self._headers = {"Authorization": f"Bearer {token}",
                         "Content-Type": "application/json"}
        # What a cost estimate is built from. Never written to results.jsonl:
        # the run manifest is a field allowlist and this is adapter-local.
        self.predict_times: list[float] = []
        self.wall_times: list[float] = []
        # Which route works, learned on the first call and then reused.
        # /v1/models/{owner}/{name}/predictions serves OFFICIAL models only and
        # 404s for community ones, which must POST /v1/predictions with a
        # version id. Detecting that by trying beats classifying models by
        # hand: it is self-correcting when Replicate promotes a model.
        self._route: str | None = None

    def _post(self, route: str, text: str) -> dict:
        body: dict = {"input": _fill(self._input, text)}
        if route == "version":
            url, body["version"] = f"{API}/predictions", self.version
        else:
            url = f"{API}/models/{self.model}/predictions"
        resp = request_with_retry(
            "POST", url, headers={**self._headers, "Prefer": "wait"},
            json=body, timeout=self._timeout)
        return resp.json()

    def _create(self, text: str) -> dict:
        if self._route:
            return self._post(self._route, text)
        order = (["model", "version"] if self.model else ["version"])
        if not self.version:
            order = ["model"]
        last: Exception | None = None
        for route in order:
            try:
                pred = self._post(route, text)
            except requests.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 404:
                    last = exc
                    continue
                raise
            self._route = route
            return pred
        raise RuntimeError(
            f"{self.id}: neither the model route nor the version route "
            f"accepted this model ({self.model or self.version}); last error "
            f"{last}") from last

    def _await(self, pred: dict) -> dict:
        """`Prefer: wait` usually returns a finished prediction, but not always
        for a cold model, so fall back to polling rather than failing."""
        deadline = time.time() + self._timeout
        while pred.get("status") in ("starting", "processing"):
            if time.time() > deadline:
                raise TimeoutError(f"{self.id}: prediction {pred.get('id')} "
                                   f"still {pred.get('status')} after "
                                   f"{self._timeout:.0f}s")
            time.sleep(self._poll_every)
            get_url = (pred.get("urls") or {}).get("get")
            if not get_url:
                raise ValueError(f"{self.id}: no polling url on {pred!r}"[:300])
            pred = request_with_retry("GET", get_url, headers=self._headers,
                                      timeout=self._timeout).json()
        return pred

    def synthesize(self, text: str, out_path: Path, sample_index: int = 0,
                   item_id: str | None = None) -> None:
        t0 = time.time()
        pred = self._await(self._create(text))
        if pred.get("status") != "succeeded":
            raise RuntimeError(f"{self.id}: prediction {pred.get('status')}: "
                               f"{str(pred.get('error'))[:200]}")

        out = pred.get("output")
        if out is None:
            raise ValueError(f"{self.id}: prediction returned no output")
        if self._output_path:
            out = dig(out, self._output_path)
        elif isinstance(out, list):
            # A model that returns a list without output_path being set is a
            # config mistake worth naming, not a silent [0].
            raise ValueError(
                f"{self.id}: output is a list of {len(out)}; set output_path "
                f"(e.g. \"0\") to say which element is the audio")
        if isinstance(out, dict):
            raise ValueError(
                f"{self.id}: output is an object with keys "
                f"{sorted(out)[:6]}; set output_path to the audio field "
                f"(bark, for instance, needs output_path: audio_out)")
        if not isinstance(out, str):
            raise ValueError(f"{self.id}: output is {type(out).__name__}, "
                             f"expected a URL string")

        # Retried like the prediction itself. An SSL or DNS blip on the
        # delivery CDN was losing a prediction that had already been paid for
        # and succeeded, which is the most expensive way to fail.
        audio = request_with_retry("GET", out, timeout=self._timeout)
        if not audio.content:
            raise ValueError(f"{self.id}: empty audio for {item_id!r}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_audio_atomic(out_path, audio.content)

        m = pred.get("metrics") or {}
        if isinstance(m.get("predict_time"), (int, float)):
            self.predict_times.append(float(m["predict_time"]))
        self.wall_times.append(time.time() - t0)
