"""Shared plumbing for the config-driven HTTP adapters.

Extracted from adapters/tts/http.py when the ASR side needed the same three
things. Secret expansion and response digging must behave identically on both
sides or a config that works for TTS fails mysteriously for ASR.

SECURITY NOTE, inherited by every caller: these take an arbitrary URL from
config. Harmless for a CLI run on your own machine. If any of this is ever
exposed as a hosted service it becomes an SSRF primitive and needs a
scheme/host allowlist plus blocking of link-local (169.254.169.254) and private
ranges. Do not host as written.
"""

from __future__ import annotations

import random
import re
import time
from typing import Any

import requests

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

# Transient by definition: a rate limit or a gateway blip is not a failed
# transcription, and recording it as one would silently deflate a system's
# denominator for reasons that have nothing to do with its speech.
RETRY_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

# A rate-limited account needs a longer leash than a healthy one, and that
# is a property of the ACCOUNT, not of the code. Six attempts at backoff 2.0
# is ~62s of waiting; seven parallel shards against a capped Replicate
# account blew through that and turned 3,202 recoverable 429s into recorded
# synthesis failures. Raise it from the environment instead of editing code:
#   TTSBENCH_RETRY_ATTEMPTS=10 TTSBENCH_RETRY_BACKOFF=3
RETRY_ATTEMPTS_ENV = "TTSBENCH_RETRY_ATTEMPTS"
RETRY_BACKOFF_ENV = "TTSBENCH_RETRY_BACKOFF"


def _env_float(name: str, default: float) -> float:
    import os
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        val = float(raw)
    except ValueError:
        return default
    return val if val > 0 else default


def expand_env(value: Any) -> Any:
    """Replace ${VAR} with the environment, recursively, raising if unset.

    Raising beats substituting an empty string: an empty Authorization header
    produces a 401 halfway through a paid run instead of at startup.
    """
    import os

    if isinstance(value, str):
        def sub(m: re.Match) -> str:
            var = m.group(1)
            got = os.environ.get(var)
            if got is None:
                raise KeyError(f"env var {var} referenced in config is not set")
            return got
        return _ENV_RE.sub(sub, value)
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    return value


def dig(obj: Any, path: str) -> Any:
    """Walk a dotted path, treating all-digit segments as list indices.

    Deepgram buries the transcript at
    results.channels.0.alternatives.0.transcript, so index support is not
    optional.
    """
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else obj[part]
    return obj



def _raise_with_body(resp: requests.Response) -> None:
    """raise_for_status(), but keep the server's explanation.

    A bare "400 Client Error: Bad Request" is the least useful thing an
    API can tell you, and it is all that reached results.jsonl for 293
    rejected clips -- the body that said WHY was thrown away at the
    moment it was received. A bounded slice costs nothing and turns the
    next such failure into a five-second diagnosis.
    """
    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        body = (resp.text or "").strip().replace("\n", " ")[:300]
        if body:
            raise requests.HTTPError(f"{exc} :: {body}",
                                     response=resp) from None
        raise

def request_with_retry(method: str, url: str, *, attempts: int | None = None,
                       backoff: float | None = None, timeout: float = 60.0,
                       sleep=time.sleep, **kwargs) -> requests.Response:
    """One HTTP call with exponential backoff and jitter on transient status.

    Honours Retry-After when the server sends it, because guessing longer than
    the server asked wastes wall clock and guessing shorter gets you banned.
    Non-transient errors (401, 400, 404) raise immediately: retrying a bad key
    five times just delays the report that the key is bad.
    """
    if attempts is None:
        attempts = int(_env_float(RETRY_ATTEMPTS_ENV, 6))
    if backoff is None:
        backoff = _env_float(RETRY_BACKOFF_ENV, 2.0)
    last: requests.Response | None = None
    for i in range(attempts):
        try:
            resp = requests.request(method, url, timeout=timeout, **kwargs)
        except requests.RequestException:
            if i == attempts - 1:
                raise
            sleep(backoff * (2 ** i) * (1 + random.random() * 0.1))
            continue
        if resp.status_code not in RETRY_STATUS:
            _raise_with_body(resp)
            return resp
        last = resp
        if i == attempts - 1:
            break
        wait = backoff * (2 ** i) * (1 + random.random() * 0.1)
        hinted = resp.headers.get("Retry-After")
        if hinted:
            try:
                wait = max(wait, float(hinted))
            except ValueError:
                pass
        sleep(wait)
    assert last is not None
    _raise_with_body(last)
    return last
