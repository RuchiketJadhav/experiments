"""Tests for zip ingestion, including hostile archives.

Run: python tests/test_ingest.py
"""

import csv
import io
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402

from audio import write_wav  # noqa: E402
from ingest import UnsafeArchive, ingest_zip, safe_extract  # noqa: E402
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


ITEMS = [
    Item("polynorm-1", "Date", "The event is on 05/20/2023.",
         "The event is on May twentieth twenty twenty three.", "polynorm"),
    Item("polynorm-2", "Currency", "The price is $10.99.",
         "The price is ten dollars and ninety nine cents.", "polynorm"),
]

tmp = Path(tempfile.mkdtemp(prefix="ttsbench-ingest-"))
try:
    wav = io.BytesIO()
    tone_path = tmp / "tone.wav"
    write_wav(tone_path, (0.2 * np.sin(np.linspace(0, 60, 8000))).astype(np.float32), 16000)
    tone = tone_path.read_bytes()

    # ---- layout A: archive carries its own manifest -----------------------
    zip_a = tmp / "with_manifest.zip"
    rows = [["index", "category", "original_text", "normalized_text",
             "model", "voice", "audio_path", "status"],
            ["1", "Date", ITEMS[0].written, ITEMS[0].spoken, "m", "v",
             r"voice\Date\1.wav", "generated"],
            ["2", "Currency", ITEMS[1].written, ITEMS[1].spoken, "m", "v",
             r"voice\Currency\2.wav", "error: upstream failed"]]
    buf = io.StringIO()
    csv.writer(buf).writerows(rows)
    with zipfile.ZipFile(zip_a, "w") as z:
        z.writestr("manifest.csv", buf.getvalue())
        z.writestr("voice/Date/1.wav", tone)
        z.writestr("voice/Currency/2.wav", tone)

    got = ingest_zip(zip_a, ITEMS, tmp / "outA", id="sysA")
    check(got.layout == "manifest", f"expected manifest layout, got {got.layout}")
    check(got.audio_files == 2, f"expected 2 audio files, got {got.audio_files}")
    check(got.matched_items == 1,
          f"the error-status row must not count as generated; got {got.matched_items}")
    check(got.adapter.resolve("polynorm-1").exists(),
          "backslash audio_path from the archive should resolve")

    # ---- layout B: no manifest, inferred from folders ---------------------
    zip_b = tmp / "bare.zip"
    with zipfile.ZipFile(zip_b, "w") as z:
        z.writestr("af_heart/Date/1.wav", tone)
        z.writestr("af_heart/Currency/2.wav", tone)
    got_b = ingest_zip(zip_b, ITEMS, tmp / "outB", id="sysB")
    check(got_b.layout == "inferred", f"expected inferred layout, got {got_b.layout}")
    check(got_b.matched_items == 2, f"both items should match, got {got_b.matched_items}")
    check(got_b.adapter.resolve("polynorm-2").exists(),
          "inferred manifest should resolve item 2")

    # ---- hostile archives -------------------------------------------------
    zip_trav = tmp / "traversal.zip"
    with zipfile.ZipFile(zip_trav, "w") as z:
        z.writestr("../escaped.wav", tone)
    raises(lambda: safe_extract(zip_trav, tmp / "outT"), UnsafeArchive,
           "`..` traversal must be rejected")

    zip_abs = tmp / "absolute.zip"
    with zipfile.ZipFile(zip_abs, "w") as z:
        z.writestr("/etc/passwd", b"x")
    raises(lambda: safe_extract(zip_abs, tmp / "outAbs"), UnsafeArchive,
           "absolute path must be rejected")

    zip_drive = tmp / "drive.zip"
    with zipfile.ZipFile(zip_drive, "w") as z:
        z.writestr("C:/windows/system32/x.wav", tone)
    raises(lambda: safe_extract(zip_drive, tmp / "outDrive"), UnsafeArchive,
           "Windows drive-letter path must be rejected")

    # entry-count cap
    import ingest as ing
    zip_many = tmp / "many.zip"
    with zipfile.ZipFile(zip_many, "w") as z:
        for i in range(12):
            z.writestr(f"a/{i}.wav", b"x")
    old = ing.MAX_ENTRIES
    ing.MAX_ENTRIES = 5
    raises(lambda: safe_extract(zip_many, tmp / "outMany"), UnsafeArchive,
           "entry-count cap must be enforced")
    ing.MAX_ENTRIES = old

    # uncompressed-size cap (a zip bomb is small on disk, huge expanded)
    zip_bomb = tmp / "bomb.zip"
    with zipfile.ZipFile(zip_bomb, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("big.bin", b"\0" * 5_000_000)
    old_b = ing.MAX_TOTAL_BYTES
    ing.MAX_TOTAL_BYTES = 1_000_000
    raises(lambda: safe_extract(zip_bomb, tmp / "outBomb"), UnsafeArchive,
           "uncompressed-size cap must be enforced")
    ing.MAX_TOTAL_BYTES = old_b

    # ---- unhelpful archives fail with a readable message ------------------
    zip_empty = tmp / "empty.zip"
    with zipfile.ZipFile(zip_empty, "w") as z:
        z.writestr("readme.txt", b"nothing here")
    raises(lambda: ingest_zip(zip_empty, ITEMS, tmp / "outE"), ValueError,
           "an archive with no audio must raise")

    zip_names = tmp / "badnames.zip"
    with zipfile.ZipFile(zip_names, "w") as z:
        z.writestr("voice/Date/first_clip.wav", tone)
    raises(lambda: ingest_zip(zip_names, ITEMS, tmp / "outN"), ValueError,
           "non-numeric filenames with no manifest must raise")

    # A manifest describing a DIFFERENT corpus must fail loudly.
    zip_wrong = tmp / "wrongtext.zip"
    bad = [rows[0], ["1", "Date", "COMPLETELY DIFFERENT TEXT.", "x", "m", "v",
                     r"voice\Date\1.wav", "generated"]]
    b2 = io.StringIO()
    csv.writer(b2).writerows(bad)
    with zipfile.ZipFile(zip_wrong, "w") as z:
        z.writestr("manifest.csv", b2.getvalue())
        z.writestr("voice/Date/1.wav", tone)
    raises(lambda: ingest_zip(zip_wrong, ITEMS, tmp / "outW"), ValueError,
           "a manifest whose text disagrees with the dataset must raise")

    print(f"{checks - len(failures)}/{checks} checks passed")
    if failures:
        print(f"\n{len(failures)} FAILED:\n")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("all passed")
finally:
    shutil.rmtree(tmp, ignore_errors=True)
