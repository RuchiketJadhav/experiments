"""Pre-synthesized audio from a folder -- the universal escape hatch.

Any TTS can be evaluated this way: synthesize offline however you like, drop
`<item_id>.wav` (optionally `<item_id>__<sample_index>.wav`) into a directory,
point the config at it. Closed systems, on-prem models and anything without a
usable API all go through here.
"""

from __future__ import annotations

import shutil
from pathlib import Path


class FolderTTS:
    def __init__(self, folder: Path | str, id: str | None = None,
                 revision: str = "n/a") -> None:
        self.folder = Path(folder)
        self.id = id or f"folder:{self.folder.name}"
        self.revision = revision

    def _source(self, item_id: str, sample_index: int) -> Path:
        candidates = [
            self.folder / f"{item_id}__{sample_index}.wav",
            self.folder / f"{item_id}.wav",
        ]
        for c in candidates:
            if c.exists():
                return c
        raise FileNotFoundError(
            f"no audio for {item_id} (sample {sample_index}) in {self.folder}"
        )

    def synthesize(self, text: str, out_path: Path, sample_index: int = 0,
                   item_id: str | None = None) -> None:
        if item_id is None:
            raise ValueError("FolderTTS needs item_id to locate pre-made audio")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._source(item_id, sample_index), out_path)
