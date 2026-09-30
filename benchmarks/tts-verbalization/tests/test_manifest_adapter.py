"""Tests for ManifestTTS against the quirks of the real manifest files.

Run: python tests/test_manifest_adapter.py
"""

import csv
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402

from adapters.base import NotGenerated  # noqa: E402
from adapters.tts.manifest import ManifestTTS  # noqa: E402
from audio import write_wav  # noqa: E402
from schema import Item  # noqa: E402

failures: list[str] = []
checks = 0


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


HEADER = ["index", "category", "original_text", "normalized_text",
          "model", "voice", "audio_path", "status"]

ROWS = [
    # backslash path, underscore category folder, spaces in dataset category
    ["1", "Biological Classification", "Homo sapiens is us.",
     "Homo sapiens is us.", "m", "v", r"voice\Biological_Classification\1.wav",
     "generated"],
    ["2", "Date", "The event is on 05/20/2023.",
     "The event is on May twentieth twenty twenty three.", "m", "v",
     r"voice\Date\2.wav", "error: Remote end closed connection"],
    ["3", "Date", "Her birthday is Oct 12.", "Her birthday is October twelfth.",
     "m", "v", r"voice\Date\3.wav", "generated"],          # file absent on disk
]

ITEMS = [
    Item("polynorm-1", "Biological Classification", "Homo sapiens is us.",
         "Homo sapiens is us.", "polynorm"),
    Item("polynorm-2", "Date", "The event is on 05/20/2023.",
         "The event is on May twentieth twenty twenty three.", "polynorm"),
    Item("polynorm-3", "Date", "Her birthday is Oct 12.",
         "Her birthday is October twelfth.", "polynorm"),
]

tmp = Path(tempfile.mkdtemp(prefix="ttsbench-man-"))
try:
    root = tmp / "set"
    (root / "voice" / "Biological_Classification").mkdir(parents=True)
    (root / "voice" / "Date").mkdir(parents=True)
    write_wav(root / "voice" / "Biological_Classification" / "1.wav",
              np.zeros(1600, dtype=np.float32), 16000)

    man = root / "manifest.csv"
    with man.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        w.writerows(ROWS)

    a = ManifestTTS("test-sys", man, root, revision="rev1")

    # Windows-style separators resolve on this platform.
    p = a.resolve("polynorm-1")
    check(p.exists(), f"backslash path should resolve to an existing file: {p}")
    check(p.name == "1.wav", f"resolved wrong file: {p}")

    # A non-generated row is excluded, not scored.
    raises(lambda: a.resolve("polynorm-2"), NotGenerated,
           "error-status row must raise NotGenerated")

    # An unknown index is also NotGenerated rather than a hard crash.
    raises(lambda: a.resolve("polynorm-99"), NotGenerated,
           "unknown index must raise NotGenerated")

    # generated_ids / categories_covered ignore the failed row.
    check(a.generated_ids() == {"polynorm-1", "polynorm-3"},
          f"generated_ids wrong: {a.generated_ids()}")
    check(a.categories_covered() == {"Biological Classification", "Date"},
          f"categories_covered wrong: {a.categories_covered()}")

    # A manifest row that points at a missing file fails per item.
    out = tmp / "out.wav"
    raises(lambda: a.synthesize("x", out, item_id="polynorm-3"),
           FileNotFoundError, "missing audio file must raise FileNotFoundError")

    # Copy-out works for a good row.
    a.synthesize("x", tmp / "ok.wav", item_id="polynorm-1")
    check((tmp / "ok.wav").exists(), "synthesize should copy the source audio")

    # Text integrity: matching dataset passes, mismatched dataset raises.
    check(a.verify_against(ITEMS) == 3, "verify_against should check 3 rows")
    bad = list(ITEMS)
    bad[0] = Item("polynorm-1", "Biological Classification",
                  "COMPLETELY DIFFERENT TEXT.", "x", "polynorm")
    raises(lambda: a.verify_against(bad), ValueError,
           "text mismatch between manifest and dataset must raise")

    # A path escaping the audio root must be refused, not silently followed.
    esc = root / "escape.csv"
    with esc.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        w.writerow(["1", "Date", "t", "t", "m", "v",
                    r"..\..\outside\1.wav", "generated"])
    b = ManifestTTS("escaper", esc, root)
    raises(lambda: b.resolve("polynorm-1"), ValueError,
           "path escaping audio_root must raise")

    # utf-8-sig: a BOM-prefixed manifest still parses (Excel writes these).
    bom = root / "bom.csv"
    bom.write_bytes(b"\xef\xbb\xbf" + man.read_bytes().split(b"\n", 1)[0]
                    + b"\n" + b",".join(x.encode() for x in ROWS[0]) + b"\n")
    c = ManifestTTS("bom", bom, root)
    check("polynorm-1" in c.rows, f"BOM manifest failed to parse: {list(c.rows)}")

    print(f"{checks - len(failures)}/{checks} checks passed")
    if failures:
        print(f"\n{len(failures)} FAILED:\n")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("all passed")
finally:
    shutil.rmtree(tmp, ignore_errors=True)
