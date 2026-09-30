"""PolyNorm-Bench loader.

Source file shape, verified against the released corpus:
    {"index": "1", "category": "Date",
     "original_text": "The event is on 05/20/2023.",
     "normalized_text": "The event is on May twentieth twenty twenty three."}

540 rows, 27 categories, 20 items each. Category names in the released file
differ from the repo README -- the file uses "Phone Number", "Fractions",
"URL or Email", "Hashtag or Mention", "Version Numbers",
"License Plate or Serial Numbers", "Sports score", "Initialism or Acronym".
Read them from the data, never hardcode the README list.

Licence: the corpus is CC BY-NC-ND 4.0. Private reformatting is permitted
("produce and reproduce, but not Share, Adapted Material"). Do not publish
rows whose source is "polynorm", nor any file derived from them.
"""

import json
from pathlib import Path
from typing import Iterator

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from schema import Item  # noqa: E402

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "raw" / "en-US_groundtruth.jsonl"


def load(path: Path | str = DEFAULT_PATH) -> list[Item]:
    return list(iter_items(path))


def iter_items(path: Path | str = DEFAULT_PATH) -> Iterator[Item]:
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                # Per-item isolation: one bad line must not kill a 540-item run.
                print(f"[polynorm] skipping malformed line {lineno}: {exc}", file=sys.stderr)
                continue
            yield Item(
                id=f"polynorm-{row['index']}",
                category=row["category"],
                written=row["original_text"],
                spoken=row["normalized_text"],
                source="polynorm",
            )


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    items = load()
    cats = sorted({i.category for i in items})
    print(f"{len(items)} items, {len(cats)} categories")
    for c in cats:
        print(" ", c)
