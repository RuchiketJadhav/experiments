"""Tests for the generic HTTP TTS adapter.

Runs against a stdlib stub server on loopback -- no network, no API key, no
provider account. Proves the config-block path works before you point it at a
paid endpoint.

Run: python tests/test_http_adapter.py
"""

import base64
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

from adapters.tts.http import HttpTTS  # noqa: E402

failures: list[str] = []
checks = 0
received: list[dict] = []


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
        failures.append(f"{label}: raised {type(exc).__name__}, wanted {exc_type.__name__}")
        return
    failures.append(f"{label}: did not raise {exc_type.__name__}")


AUDIO = b"RIFF____WAVEfmt fake audio payload"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):        # keep the test output clean
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        received.append({"path": self.path, "body": body,
                         "headers": dict(self.headers)})
        if self.path == "/raw":
            self.send_response(200); self.send_header("Content-Type", "audio/wav")
            self.end_headers(); self.wfile.write(AUDIO)
        elif self.path == "/b64":
            payload = json.dumps(
                {"audio": {"data": base64.b64encode(AUDIO).decode()}}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers(); self.wfile.write(payload)
        elif self.path == "/empty":
            self.send_response(200); self.send_header("Content-Length", "0")
            self.end_headers()
        else:
            self.send_response(500); self.end_headers(); self.wfile.write(b"boom")


srv = HTTPServer(("127.0.0.1", 0), Handler)
port = srv.server_port
threading.Thread(target=srv.serve_forever, daemon=True).start()
tmp = Path(tempfile.mkdtemp(prefix="ttsbench-http-"))
try:
    base = f"http://127.0.0.1:{port}"
    os.environ["TTSBENCH_FAKE_KEY"] = "secret-token-123"

    # 1. Raw audio body + {{text}} templating + ${ENV} header expansion.
    a = HttpTTS({
        "id": "stub-raw", "revision": "v1", "endpoint": f"{base}/raw",
        "headers": {"Authorization": "Bearer ${TTSBENCH_FAKE_KEY}"},
        "body": {"text": "{{text}}", "model": "stub-1", "sample_rate": 16000},
        "response": "audio",
    })
    out = tmp / "raw.wav"
    a.synthesize("The price is $10.99.", out, item_id="polynorm-1")
    check(out.exists() and out.read_bytes() == AUDIO,
          "raw response body should be written verbatim to the wav path")
    check(received[-1]["body"]["text"] == "The price is $10.99.",
          f"{{{{text}}}} should be substituted; got {received[-1]['body'].get('text')!r}")
    check(received[-1]["body"]["model"] == "stub-1",
          "non-template body fields should pass through unchanged")
    check(received[-1]["headers"].get("Authorization") == "Bearer secret-token-123",
          "${ENV} in headers should expand to the environment value")
    check(a.id == "stub-raw" and a.revision == "v1",
          "adapter should carry id and revision for the manifest")

    # 2. Base64 payload dug out of a JSON response.
    b = HttpTTS({
        "id": "stub-b64", "endpoint": f"{base}/b64",
        "body": {"text": "{{text}}"},
        "response": {"json_path": "audio.data", "encoding": "base64"},
    })
    out2 = tmp / "b64.wav"
    b.synthesize("hello", out2, item_id="polynorm-2")
    check(out2.read_bytes() == AUDIO,
          "base64 payload at a json_path should decode to the audio bytes")

    # 3. An empty 200 must fail loudly, not write a zero-byte wav that later
    #    looks like a TTS failure.
    c = HttpTTS({"id": "stub-empty", "endpoint": f"{base}/empty",
                 "body": {"text": "{{text}}"}})
    raises(lambda: c.synthesize("x", tmp / "empty.wav", item_id="polynorm-3"),
           ValueError, "empty 200 response must raise")
    check(not (tmp / "empty.wav").exists(),
          "a failed call must not leave a partial file behind")

    # 4. Non-2xx propagates (run.py isolates it per item).
    d = HttpTTS({"id": "stub-500", "endpoint": f"{base}/oops",
                 "body": {"text": "{{text}}"}})
    raises(lambda: d.synthesize("x", tmp / "err.wav", item_id="polynorm-4"),
           Exception, "HTTP 500 must raise")

    # 5. A missing env var is caught at construction, naming the variable.
    os.environ.pop("TTSBENCH_MISSING", None)
    try:
        HttpTTS({"id": "x", "endpoint": base,
                 "headers": {"Authorization": "Bearer ${TTSBENCH_MISSING}"},
                 "body": {}})
        failures.append("missing env var should raise")
        checks += 1
    except KeyError as exc:
        checks += 1
        check("TTSBENCH_MISSING" in str(exc),
              f"error should name the missing variable; got {exc}")

    print(f"{checks - len(failures)}/{checks} checks passed")
    if failures:
        print(f"\n{len(failures)} FAILED:\n")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("all passed")
finally:
    srv.shutdown()
    shutil.rmtree(tmp, ignore_errors=True)
