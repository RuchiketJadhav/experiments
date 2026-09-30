"""Stratified sampling across categories.

WHY THIS EXISTS
`--limit N` takes the first N dataset rows. PolyNorm is ordered by category, so
`--limit 10` is ten Date items and nothing else -- every number produced before
this module existed was a date score wearing a general label.

`sample_items` spreads the sample round-robin over all 27 categories instead, so
a 20-item run tells you something about the corpus rather than about dates. It
is deterministic under a fixed seed so two runs stay comparable.
"""

from __future__ import annotations

import random
from collections import OrderedDict

from schema import Item


def sample_items(items: list[Item], n: int, seed: int = 0) -> list[Item]:
    """Up to `n` items spread as evenly as possible across categories.

    Round-robin, so with 27 categories a 20-item sample hits 20 distinct
    categories rather than one. Returns dataset order for stable output.
    """
    if n <= 0 or n >= len(items):
        return list(items)

    by_cat: "OrderedDict[str, list[Item]]" = OrderedDict()
    for it in items:
        by_cat.setdefault(it.category, []).append(it)

    rng = random.Random(seed)
    pools = []
    for cat in by_cat:
        pool = list(by_cat[cat])
        rng.shuffle(pool)
        pools.append(pool)
    # Rotate the category order too, so a small n does not always favour the
    # categories that happen to sort first.
    rng.shuffle(pools)

    picked: list[Item] = []
    depth = 0
    while len(picked) < n:
        progressed = False
        for pool in pools:
            if depth < len(pool):
                picked.append(pool[depth])
                progressed = True
                if len(picked) == n:
                    break
        if not progressed:
            break
        depth += 1

    order = {id(it): i for i, it in enumerate(items)}
    picked.sort(key=lambda it: order[id(it)])
    return picked
