"""Turn an uploaded zip of audio into a scoring adapter.

Two layouts are accepted, both matching what real TTS export runs produce:

  A. Contains manifest.csv (or manifest_*.csv) -> handed straight to ManifestTTS,
     which already deals with backslash paths, underscore category folders,
     8/9-column headers, BOM and non-"generated" status rows.

  B. No manifest, just <anything>/<Category>/<index>.wav -> a manifest is
     SYNTHESIZED from the dataset and the file layout, then handed to the same
     adapter. One code path does the scoring either way.

Extraction is hostile-input aware: absolute paths, `..` traversal, entry-count
and uncompressed-size caps are all enforced before anything is written. The user
running this locally is not the threat model; a zip downloaded from someone else
is.
"""

from __future__ import annotations

import csv
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from adapters.tts.manifest import ManifestTTS
from schema import Item

MAX_ENTRIES = 5_000
MAX_TOTAL_BYTES = 1_500_000_000        # 1.5 GB uncompressed
AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a"}


class UnsafeArchive(Exception):
    """The archive tried something an archive should not."""


@dataclass
class Ingested:
    adapter: ManifestTTS
    layout: str            # "manifest" | "inferred"
    audio_files: int
    matched_items: int
    root: Path


def safe_extract(zip_path: Path | str, dest: Path | str) -> Path:
    dest = Path(dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        infos = [i for i in zf.infolist() if not i.is_dir()]
        if len(infos) > MAX_ENTRIES:
            raise UnsafeArchive(f"{len(infos)} entries exceeds the {MAX_ENTRIES} cap")
        total = sum(i.file_size for i in infos)
        if total > MAX_TOTAL_BYTES:
            raise UnsafeArchive(
                f"uncompressed size {total/1e9:.1f} GB exceeds the "
                f"{MAX_TOTAL_BYTES/1e9:.1f} GB cap")
        for info in infos:
            name = info.filename.replace("\\", "/")
            p = PurePosixPath(name)
            if p.is_absolute() or ".." in p.parts or (len(name) > 1 and name[1] == ":"):
                raise UnsafeArchive(f"unsafe path in archive: {info.filename!r}")
            out = (dest / Path(*p.parts)).resolve()
            if not out.is_relative_to(dest):
                raise UnsafeArchive(f"path escapes extraction root: {info.filename!r}")
            out.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, out.open("wb") as dst:
                dst.write(src.read())
    return dest


def _find_manifest(root: Path) -> Path | None:
    hits = sorted(p for p in root.rglob("*.csv")
                  if p.name.lower().startswith("manifest"))
    return hits[0] if hits else None


def _audio_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in AUDIO_EXT)


def _synthesize_manifest(root: Path, items: list[Item], audio: list[Path]) -> Path:
    """Build a manifest from a bare <category>/<index>.<ext> tree.

    The index comes from the filename stem; text comes from the dataset, so the
    generated manifest cannot disagree with what is being scored.
    """
    by_index = {}
    for p in audio:
        stem = p.stem.split("__")[0]
        if stem.isdigit():
            by_index.setdefault(stem, p)
    if not by_index:
        raise ValueError(
            "no manifest.csv, and no <category>/<index>.wav files were found. "
            "Name files by their dataset index (1.wav, 2.wav, ...) or include a "
            "manifest.csv.")

    out = root / "_generated_manifest.csv"
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["index", "category", "original_text", "normalized_text",
                    "model", "voice", "audio_path", "status"])
        for item in items:
            idx = item.id.split("-")[-1]
            src = by_index.get(idx)
            if src is None:
                continue
            w.writerow([idx, item.category, item.written, item.spoken,
                        "uploaded", "uploaded",
                        str(src.relative_to(root)), "generated"])
    return out


def ingest_zip(zip_path: Path | str, items: list[Item], workdir: Path | str,
               id: str = "uploaded", revision: str = "upload") -> Ingested:
    root = safe_extract(zip_path, workdir)
    audio = _audio_files(root)
    if not audio:
        raise ValueError("the archive contains no audio files")

    manifest = _find_manifest(root)
    layout = "manifest"
    if manifest is None:
        manifest = _synthesize_manifest(root, items, audio)
        layout = "inferred"

    # audio_path values are relative to the manifest's own directory.
    adapter = ManifestTTS(id, manifest, manifest.parent, revision=revision)
    if layout == "manifest":
        # A real manifest may describe a different corpus; fail loudly here
        # rather than silently score one model's audio under another's text.
        adapter.verify_against(items)

    ids = {i.id for i in items}
    matched = len(adapter.generated_ids() & ids)
    if matched == 0:
        raise ValueError(
            "none of the archive's rows match the dataset. Check that the "
            "index column lines up with the PolyNorm indices.")
    return Ingested(adapter=adapter, layout=layout, audio_files=len(audio),
                    matched_items=matched, root=root)
