"""HTTP ASR adapter guards.

The properties that matter are the ones that would otherwise corrupt a paid run
without anyone noticing:

- a rate limit must be RETRIED, never recorded as a failed transcription, or a
  429 storm silently deflates a system's denominator
- a bad key must fail immediately rather than five times slowly
- a recognizer must DECLARE whether it rewrites speech into written form; that
  flag is what stands between the report and an ITN false positive
- a key must never end up anywhere it could be written to disk

Run: python tests/test_http_asr.py
"""

import json
import os
import shutil
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

import requests  # noqa: E402

from adapters.asr.http import HttpASR  # noqa: E402
from adapters.httputil import dig, expand_env, request_with_retry  # noqa: E402

failures: list[str] = []
checks = 0
received: list[dict] = []
hits: dict[str, int] = {}


def check(cond: bool, label: str) -> None:
    global checks
    checks += 1
    if not cond:
        failures.append(label)


def raises(fn, exc_type, label: str) -> None:
    global checks
    checks += 1
    try:
        fn()
    except exc_type:
        return
    except Exception as exc:
        failures.append(f"{label}: raised {type(exc).__name__}, "
                        f"wanted {exc_type.__name__}")
        return
    failures.append(f"{label}: did not raise {exc_type.__name__}")


WAV = b"RIFF____WAVEfmt " + b"\x00" * 64


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, payload: bytes, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n)
        path = self.path.split("?")[0]
        hits[path] = hits.get(path, 0) + 1
        received.append({"path": self.path, "raw": raw,
                         "headers": dict(self.headers)})

        if path == "/openai":                       # multipart, flat json
            self._send(200, json.dumps({"text": "The event is on May 20th."}).encode())
        elif path == "/deepgram":                   # raw body, nested json + index
            self._send(200, json.dumps({"results": {"channels": [
                {"alternatives": [{"transcript": "may twentieth"}]}]}}).encode())
        elif path == "/plain":
            self._send(200, b"  bare text body  ", "text/plain")
        elif path == "/badshape":
            self._send(200, json.dumps({"unexpected": True}).encode())
        elif path == "/nonstring":
            self._send(200, json.dumps({"text": {"nested": 1}}).encode())
        elif path == "/flaky":                      # 429 twice, then success
            if hits[path] <= 2:
                self.send_response(429)
                self.send_header("Retry-After", "0")
                self.send_header("Content-Length", "0")
                self.end_headers()
            else:
                self._send(200, json.dumps({"text": "recovered"}).encode())
        elif path == "/always429":          # never recovers, for backoff maths
            self.send_response(429)
            self.send_header("Retry-After", "0")
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif path == "/badkey":
            self._send(401, json.dumps({"error": "bad key"}).encode())
        else:
            self._send(404, b"{}")


srv = HTTPServer(("127.0.0.1", 0), Handler)
threading.Thread(target=srv.serve_forever, daemon=True).start()
BASE = f"http://127.0.0.1:{srv.server_address[1]}"

tmp = Path(tempfile.mkdtemp(prefix="ttsasr-"))
try:
    wav = tmp / "clip.wav"
    wav.write_bytes(WAV)
    os.environ["TEST_ASR_KEY"] = "sk-secret-never-log-me"

    # ------------------------------------------------- multipart (OpenAI shape)
    a = HttpASR({"id": "openai-whisper-1", "type": "http", "upload": "multipart",
                 "normalizes_output": True, "endpoint": f"{BASE}/openai",
                 "headers": {"Authorization": "Bearer ${TEST_ASR_KEY}"},
                 "form": {"model": "whisper-1", "response_format": "json"},
                 "json_path": "text"})
    check(a.transcribe(wav) == "The event is on May 20th.", "multipart transcript")
    sent = received[-1]
    check(b"filename=" in sent["raw"], "multipart must upload the wav as a file part")
    check(WAV in sent["raw"], "the wav bytes must reach the server intact")
    check(b"whisper-1" in sent["raw"], "form fields must be sent alongside the file")
    check(sent["headers"].get("Authorization") == "Bearer sk-secret-never-log-me",
          "${ENV} must expand in headers")
    check(a.normalizes_output is True, "declared normalizes_output must be kept")

    # --------------------------------------------------- raw body (Deepgram)
    d = HttpASR({"id": "deepgram-nova-3", "type": "http", "upload": "raw",
                 "normalizes_output": False,
                 "endpoint": f"{BASE}/deepgram?smart_format=false&numerals=false",
                 "headers": {"Authorization": "Token ${TEST_ASR_KEY}",
                             "Content-Type": "audio/wav"},
                 "json_path": "results.channels.0.alternatives.0.transcript"})
    check(d.transcribe(wav) == "may twentieth", "nested json_path with list index")
    check(received[-1]["raw"] == WAV,
          "raw upload must send the wav as the body, unwrapped")
    check("numerals=false" in received[-1]["path"],
          "the formatting-off query string must survive to the request")
    check(d.normalizes_output is False, "non-normalizing must be recorded as such")

    # -------------------------------------------------------- response shapes
    p = HttpASR({"id": "plain", "normalizes_output": False, "upload": "raw",
                 "endpoint": f"{BASE}/plain"})
    check(p.transcribe(wav) == "bare text body", "no json_path -> stripped body")

    bad = HttpASR({"id": "bad", "normalizes_output": False, "upload": "raw",
                   "endpoint": f"{BASE}/badshape", "json_path": "results.0.text"})
    try:
        bad.transcribe(wav)
        checks += 1
        failures.append("a wrong json_path must raise")
    except ValueError as exc:
        checks += 1
        check("results.0.text" in str(exc),
              "the error must name the path that failed")
        check("unexpected" in str(exc), "the error must show part of the body")

    ns = HttpASR({"id": "ns", "normalizes_output": False, "upload": "raw",
                  "endpoint": f"{BASE}/nonstring", "json_path": "text"})
    raises(lambda: ns.transcribe(wav), ValueError, "a non-string transcript must raise")

    # ------------------------------------------------------------- retries
    f = HttpASR({"id": "flaky", "normalizes_output": False, "upload": "raw",
                 "endpoint": f"{BASE}/flaky", "json_path": "text", "attempts": 5})
    check(f.transcribe(wav) == "recovered",
          "a 429 must be retried, not reported as a failed transcription")
    check(hits["/flaky"] == 3, f"expected 3 attempts, got {hits.get('/flaky')}")

    before = hits.get("/badkey", 0)
    k = HttpASR({"id": "badkey", "normalizes_output": False, "upload": "raw",
                 "endpoint": f"{BASE}/badkey", "json_path": "text"})
    raises(lambda: k.transcribe(wav), requests.HTTPError, "401 must raise")
    check(hits["/badkey"] - before == 1,
          "a bad key must fail on the first attempt, not be retried")

    # Backoff maths, without waiting for it. Uses a route that NEVER recovers:
    # pointing this at /flaky made it order-dependent, since the earlier test
    # had already consumed that route's two 429s.
    waits: list[float] = []
    raises(lambda: request_with_retry(
        "POST", f"{BASE}/always429", attempts=3, backoff=1.0,
        sleep=waits.append, data=b"x"), requests.HTTPError,
        "exhausted retries must raise the last response")
    check(len(waits) == 2, f"attempts=3 should sleep twice, slept {len(waits)}")
    check(hits["/always429"] == 3,
          f"attempts=3 should hit 3 times, got {hits.get('/always429')}")

    # ---------------------------------------------------------- config guards
    raises(lambda: HttpASR({"id": "x", "endpoint": BASE}), ValueError,
           "a recognizer that does not declare normalizes_output must be rejected")
    raises(lambda: HttpASR({"id": "x", "endpoint": BASE, "normalizes_output": False,
                            "upload": "sftp"}), ValueError,
           "an unknown upload mode must be rejected")
    raises(lambda: expand_env("${DEFINITELY_NOT_SET_XYZ}"), KeyError,
           "a missing env var must raise, not expand to empty")

    check(dig({"a": [{"b": "c"}]}, "a.0.b") == "c", "dig walks list indices")

    # ------------------------------------------------------------- key safety
    # Whatever the adapter keeps must not be serialisable back into a run record.
    blob = json.dumps({k: str(v) for k, v in vars(a).items()
                       if not k.startswith("_")}).lower()
    check("sk-secret" not in blob,
          "no public attribute may carry the key into a record")
    check("sk-secret" not in f"{a.id}{a.revision}",
          "the id and revision are written to results.jsonl and must stay clean")

    # ------------------------------------------------ build_asr honours enabled
    # build_tts skipped disabled blocks but build_asr did not, so a config
    # listing two API recognizers was unloadable until you held BOTH keys --
    # the same defect that once made api_example.yaml unusable.
    from config import build_asr  # noqa: E402
    cfg = {"asr": [
        {"type": "wav2vec2", "model": "facebook/wav2vec2-base-960h"},
        {"id": "off", "type": "http", "enabled": False, "upload": "raw",
         "normalizes_output": False, "endpoint": "https://example.invalid",
         "headers": {"Authorization": "Token ${DEFINITELY_NOT_SET_XYZ}"}},
    ]}
    built = build_asr(cfg)
    check(len(built) == 1,
          f"a disabled recognizer must be skipped, built {len(built)}")
    check("off" not in [x.id for x in built], "the disabled id must not be built")

    on = dict(cfg["asr"][1]); on.pop("enabled")
    raises(lambda: build_asr({"asr": [on]}), KeyError,
           "an ENABLED block with a missing key must fail loudly at load")

    print(f"{checks - len(failures)}/{checks} checks passed")
    if failures:
        print(f"\n{len(failures)} FAILED:\n")
        for x in failures:
            print(f"  - {x}")
        sys.exit(1)
    print("all passed")
finally:
    srv.shutdown()
    shutil.rmtree(tmp, ignore_errors=True)
