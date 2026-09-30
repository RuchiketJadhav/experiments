"""Synthetic corpus loader.

Mirrors data/loaders/polynorm.py exactly, so nothing downstream can tell the
two apart: same Item shape, same category names, same results.jsonl. The only
difference is `source`, which is what makes provenance auditable when a run
mixes sets -- PolyNorm rows may not be redistributed, these rows may.

File shape, one JSON object per line, written by gen_corpus.py:
    {"id": "syn-en-date-001", "category": "Date",
     "written": "The event is scheduled for 05/20/2023 at the main hall.",
     "spoken":  "The event is scheduled for May twentieth twenty twenty "
                "three at the main hall.",
     "fragment_written": "05/20/2023",
     "fragment_spoken": "May twentieth twenty twenty three",
     "writeback_scores": true}

Licence: generated here from data/spec/*.yaml and a seed. No third-party
rights attach, so the corpus, the code and the results publish together.
"""

import json
import sys
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from schema import Item  # noqa: E402

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "synthetic" / "en-US.jsonl"


def load(path: Path | str = DEFAULT_PATH) -> list[Item]:
    return list(iter_items(path))


def iter_items(path: Path | str = DEFAULT_PATH) -> Iterator[Item]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- generate it first: python gen_corpus.py")
    locale = path.stem
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                # Per-item isolation: one bad line must not kill the run.
                print(f"[synthetic] skipping malformed line {lineno}: {exc}",
                      file=sys.stderr)
                continue
            yield Item(
                id=row["id"],
                category=row["category"],
                written=row["written"],
                spoken=row["spoken"],
                source=f"synthetic-{locale}",
            )


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    items = load()
    from collections import Counter
    counts = Counter(i.category for i in items)
    print(f"{len(items)} items, {len(counts)} categories")
    for c, n in sorted(counts.items()):
        print(f"  {n:3d}  {c}")
