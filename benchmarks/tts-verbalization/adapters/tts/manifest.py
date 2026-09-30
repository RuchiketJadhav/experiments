"""Serve pre-synthesized audio described by a manifest CSV.

Written against the real files in Downloads, whose shape varies:

  index,category,original_text,normalized_text[,source],model,voice,audio_path,status

- `audio_path` uses Windows backslashes: "af_heart\\Date\\1.wav".
- Category folders use underscores ("Biological_Classification") while PolyNorm
  categories use spaces, so paths are ALWAYS resolved through the manifest's own
  audio_path and never rebuilt from a category name.
- Some sets have a `source` column and some do not; fields are read by name.
- `status` is "generated" or an error string. Non-generated rows raise
  NotGenerated so they are excluded rather than scored.
"""

from __future__ import annotations

import csv
from pathlib import Path, PureWindowsPath

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from adapters.base import NotGenerated  # noqa: E402
from schema import Item  # noqa: E402


class ManifestTTS:
    def __init__(self, id: str, manifest: Path | str, audio_root: Path | str,
                 revision: str = "unknown") -> None:
        self.id = id
        self.revision = revision
        self.manifest_path = Path(manifest)
        self.audio_root = Path(audio_root).resolve()

        self.rows: dict[str, dict] = {}
        with self.manifest_path.open(encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                idx = (row.get("index") or "").strip()
                if not idx:
                    continue
                self.rows[f"polynorm-{idx}"] = row

    # -- inspection ---------------------------------------------------------
    def generated_ids(self) -> set[str]:
        return {k for k, r in self.rows.items()
                if (r.get("status") or "").strip() == "generated"}

    def categories_covered(self) -> set[str]:
        return {r.get("category", "") for k, r in self.rows.items()
                if k in self.generated_ids()}

    def resolve(self, item_id: str) -> Path:
        row = self.rows.get(item_id)
        if row is None:
            raise NotGenerated(f"{self.id}: no manifest row for {item_id}")
        status = (row.get("status") or "").strip()
        if status != "generated":
            raise NotGenerated(f"{self.id}: status={status[:60]!r}")
        rel = PureWindowsPath((row.get("audio_path") or "").strip())
        if not rel.parts:
            raise NotGenerated(f"{self.id}: empty audio_path for {item_id}")
        path = (self.audio_root / Path(*rel.parts)).resolve()
        # A wrong manifest/audio_root pairing must fail loudly, not silently
        # score one model's audio under another model's label.
        if not path.is_relative_to(self.audio_root):
            raise ValueError(f"{self.id}: {path} escapes audio root {self.audio_root}")
        return path

    def verify_against(self, items: list[Item]) -> int:
        """Manifest text must equal the dataset text for the same index.

        Returns the number of rows checked. Raises on the first mismatch --
        audio that does not correspond to the scored text invalidates every
        number downstream, and does so invisibly.
        """
        checked = 0
        by_id = {i.id: i for i in items}
        for item_id, row in self.rows.items():
            item = by_id.get(item_id)
            if item is None:
                continue
            got = (row.get("original_text") or "").strip()
            if got and got != item.written.strip():
                raise ValueError(
                    f"{self.id}: text mismatch at {item_id}\n"
                    f"  manifest: {got!r}\n  dataset : {item.written!r}"
                )
            checked += 1
        return checked

    # -- TTSAdapter ---------------------------------------------------------
    def synthesize(self, text: str, out_path: Path, sample_index: int = 0,
                   item_id: str | None = None) -> None:
        if item_id is None:
            raise ValueError("ManifestTTS needs item_id")
        src = self.resolve(item_id)
        if not src.exists():
            raise FileNotFoundError(f"{self.id}: manifest points at missing {src}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(src.read_bytes())
