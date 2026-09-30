"""Internal item schema.

Every dataset loader maps onto this shape so the rest of the pipeline never
knows which corpus a row came from. `source` keeps provenance auditable when
sets are mixed -- PolyNorm-derived rows carry redistribution constraints that
self-authored rows do not.
"""

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Item:
    id: str
    category: str
    written: str   # fed to the TTS
    spoken: str    # the reference verbalization
    source: str    # "polynorm" | "mine" | ...

    def as_dict(self) -> dict:
        return asdict(self)
