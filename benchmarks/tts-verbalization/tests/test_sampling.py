"""Tests for stratified sampling.

Run: python tests/test_sampling.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from sampling import sample_items  # noqa: E402

# Stratified sampling is a property of ANY category-ordered corpus, so this
# runs against whichever corpus is present. PolyNorm first, because it is
# the corpus the sampling bug was found on -- but it is CC BY-NC-ND and
# therefore absent from the cloud deployment, where the suite used to die
# with FileNotFoundError and fail an otherwise healthy bootstrap.
_POLY = (Path(__file__).resolve().parents[1]
         / "data" / "raw" / "en-US_groundtruth.jsonl")
if _POLY.exists():
    from data.loaders.polynorm import load  # noqa: E402
    CORPUS = "polynorm"
else:
    from data.loaders.synthetic import load  # noqa: E402
    CORPUS = "synthetic"

failures: list[str] = []
checks = 0


def check(cond: bool, label: str) -> None:
    global checks
    checks += 1
    if not cond:
        failures.append(label)


items = load()
n_cats = len({i.category for i in items})
check(n_cats == 27, f"expected 27 categories, got {n_cats}")

# The bug this module exists to fix: the first 10 rows are all one category.
first10 = items[:10]
check(len({i.category for i in first10}) == 1,
      "sanity: --limit 10 really is single-category (that was the bug)")

# 27 items -> 27 distinct categories.
s27 = sample_items(items, 27, seed=0)
check(len(s27) == 27, f"expected 27 items, got {len(s27)}")
check(len({i.category for i in s27}) == 27,
      f"27-item sample should cover 27 categories, got {len({i.category for i in s27})}")

# 20 items -> 20 distinct categories (round-robin, never doubles up early).
s20 = sample_items(items, 20, seed=0)
check(len(s20) == 20, f"expected 20 items, got {len(s20)}")
check(len({i.category for i in s20}) == 20,
      f"20-item sample should hit 20 categories, got {len({i.category for i in s20})}")

# 54 items -> every category represented, twice each.
s54 = sample_items(items, 54, seed=0)
check(len({i.category for i in s54}) == 27, "54-item sample should cover all categories")

# Deterministic under a seed; different seeds give different draws.
check([i.id for i in sample_items(items, 20, seed=0)] == [i.id for i in s20],
      "same seed must reproduce the same sample")
check([i.id for i in sample_items(items, 20, seed=1)] != [i.id for i in s20],
      "a different seed should draw a different sample")

# Degenerate inputs.
check(len(sample_items(items, 10_000)) == len(items), "n > len(items) returns everything")
check(sample_items(items, 0) == items, "n <= 0 returns everything")
check(sample_items([], 5) == [], "empty input is safe")

# Output stays in dataset order, so results files read predictably.
idx = {it.id: n for n, it in enumerate(items)}
check(all(idx[a.id] < idx[b.id] for a, b in zip(s20, s20[1:])),
      "sample should be returned in dataset order")

# No duplicates.
check(len({i.id for i in s54}) == len(s54), "sample must not repeat an item")

print(f"{checks - len(failures)}/{checks} checks passed [{CORPUS} corpus]")
if failures:
    print(f"\n{len(failures)} FAILED:\n")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("all passed")
