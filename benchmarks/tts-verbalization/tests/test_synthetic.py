"""Tests for the generated corpus and the loader that feeds it to a run.

Two jobs. First, the number machinery in gen_corpus.py is the reference
reading for every item -- if cardinal() is wrong, the benchmark is measuring
the generator's arithmetic rather than the voice, and nothing downstream would
ever notice. Second, the loader has to be indistinguishable from the PolyNorm
loader, because the whole point of the swap is that score.py, report.py and
audit.py keep working untouched.

Run: python tests/test_synthetic.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

import yaml  # noqa: E402

from canonicalize import has_target, target_match  # noqa: E402
from gen_corpus import (GENERATORS, alnum_words, cardinal,  # noqa: E402
                        decimal_words, digit_words, generate, ordinal_digits,
                        ordinal_words, roman, verify, year_words)
from schema import Item  # noqa: E402

failures: list[str] = []
checks = 0


def check(label: str, got, want) -> None:
    global checks
    checks += 1
    if got != want:
        failures.append(f"{label}: got {got!r}, want {want!r}")


def check_true(label: str, got) -> None:
    check(label, bool(got), True)


# --------------------------------------------------------------------------
# Number words. These are the reference readings; an error here is invisible
# downstream because every model would be scored against the same wrong answer.
# --------------------------------------------------------------------------

for n, want in [
    (0, "zero"), (7, "seven"), (13, "thirteen"), (20, "twenty"),
    (21, "twenty one"), (99, "ninety nine"), (100, "one hundred"),
    (118, "one hundred eighteen"), (900, "nine hundred"),
    (1000, "one thousand"), (1234, "one thousand two hundred thirty four"),
    (1_000_000, "one million"),
    (1_234_567, "one million two hundred thirty four thousand five hundred sixty seven"),
    (8688, "eight thousand six hundred eighty eight"),
]:
    check(f"cardinal({n})", cardinal(n), want)

for n, want in [
    (1, "first"), (2, "second"), (3, "third"), (4, "fourth"), (5, "fifth"),
    (8, "eighth"), (9, "ninth"), (12, "twelfth"), (13, "thirteenth"),
    (20, "twentieth"), (21, "twenty first"), (40, "fortieth"),
    (47, "forty seventh"), (100, "one hundredth"), (103, "one hundred third"),
]:
    check(f"ordinal_words({n})", ordinal_words(n), want)

for n, want in [(1, "1st"), (2, "2nd"), (3, "3rd"), (4, "4th"), (11, "11th"),
                (12, "12th"), (13, "13th"), (21, "21st"), (22, "22nd"),
                (23, "23rd"), (101, "101st"), (111, "111th")]:
    check(f"ordinal_digits({n})", ordinal_digits(n), want)

for y, want in [
    (1995, "nineteen ninety five"), (2023, "twenty twenty three"),
    (2000, "two thousand"), (2005, "two thousand five"),
    (1905, "nineteen oh five"), (1900, "nineteen hundred"),
    (2035, "twenty thirty five"), (2010, "twenty ten"),
]:
    check(f"year_words({y})", year_words(y), want)

check("digit_words", digit_words("555-0199"), "five five five zero one nine nine")
check("decimal_words", decimal_words(3, "14159"),
      "three point one four one five nine")
check("alnum_words", alnum_words("ABC-1234"), "A B C one two three four")
check("alnum_words formula", alnum_words("H2O"), "H two O")

for n, want in [(4, "IV"), (7, "VII"), (9, "IX"), (14, "XIV"), (40, "XL"),
                (56, "LVI"), (90, "XC"), (400, "CD")]:
    check(f"roman({n})", roman(n), want)


# --------------------------------------------------------------------------
# The spec and the generator table must agree, or a category silently vanishes.
# --------------------------------------------------------------------------

spec_path = Path(__file__).resolve().parents[1] / "data" / "spec" / "en.yaml"
spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
spec_cats = set(spec["categories"])
check("every spec category has a generator", spec_cats - set(GENERATORS), set())
check("no orphan generators", set(GENERATORS) - spec_cats, set())

for cat, entry in spec["categories"].items():
    check_true(f"{cat} has carriers", entry.get("carriers"))
    for carrier in entry["carriers"]:
        check(f"{cat} carrier has exactly one slot", carrier.count("{x}"), 1)
        # A bare fragment trips screen.py's 0.60 s-per-character rule and
        # throws out correct audio in the high-expansion categories.
        check_true(f"{cat} carrier is a sentence: {carrier!r}",
                   len(carrier) >= 30 and carrier.rstrip().endswith("."))


# --------------------------------------------------------------------------
# Generation is deterministic, and every emitted row is scorable.
# --------------------------------------------------------------------------

def run(seed: int, count: int) -> list[dict]:
    return [row for row, _ in generate(spec, seed, count) if row]


a = run(1234, 3)
b = run(1234, 3)
check("same seed gives the same corpus", a, b)
c = run(9999, 3)
check_true("a different seed gives a different corpus", a != c)

check("ids are unique", len({r["id"] for r in a}), len(a))
for row in a:
    cat = row["category"]
    check_true(f"{row['id']} has a target",
               has_target(row["written"], row["spoken"], cat))
    check_true(f"{row['id']} accepts its own reference reading",
               target_match(row["spoken"], row["written"], row["spoken"], cat))
    check_true(f"{row['id']} written text is non-empty", row["written"].strip())
    check_true(f"{row['id']} spoken text is non-empty", row["spoken"].strip())

# verify() must actually reject: a form that says nothing new has no target.
reason, _ = verify("The total is 12 dollars.", "The total is 12 dollars.",
                   "12 dollars", "12 dollars", "Currency")
check("verify rejects a no-target item", reason, "no_target")


# --------------------------------------------------------------------------
# The loader. Must be interchangeable with the PolyNorm one.
# --------------------------------------------------------------------------

corpus = Path(__file__).resolve().parents[1] / "data" / "synthetic" / "en-US.jsonl"
if not corpus.exists():
    print(f"SKIP loader checks: {corpus} not generated yet")
else:
    from data.loaders.synthetic import load  # noqa: E402
    items = load(corpus)
    check_true("loader returns items", items)
    check_true("loader yields Item", all(isinstance(i, Item) for i in items))
    check("source records provenance",
          {i.source for i in items}, {"synthetic-en-US"})
    check("no empty written text",
          [i.id for i in items if not i.written.strip()], [])
    check("no empty spoken text",
          [i.id for i in items if not i.spoken.strip()], [])
    check("ids unique on disk", len({i.id for i in items}), len(items))

    # Category names must match PolyNorm's exactly, or per-category results
    # cannot be lined up against everything already measured.
    raw = [json.loads(line) for line in
           corpus.read_text(encoding="utf-8").splitlines() if line.strip()]
    counts = Counter(r["category"] for r in raw)
    check("all 27 categories present", len(counts), 27)
    target = spec.get("default_count", 40)
    short = {c: n for c, n in counts.items() if n != target}
    check(f"every category at {target} items", short, {})

    poly = Path(__file__).resolve().parents[1] / "data" / "raw" / "en-US_groundtruth.jsonl"
    if poly.exists():
        poly_cats = {json.loads(line)["category"] for line in
                     poly.read_text(encoding="utf-8").splitlines() if line.strip()}
        check("category names match PolyNorm exactly", set(counts), poly_cats)

    # Regeneration must be stable: the ids downstream records point at cannot
    # move, or a resumed run would append duplicates under new names.
    again = run(20260923, target)
    check("regeneration reproduces the ids on disk",
          [r["id"] for r in again], [r["id"] for r in raw])
    check("regeneration reproduces the text on disk",
          [r["written"] for r in again], [r["written"] for r in raw])

print(f"{checks - len(failures)}/{checks} checks passed (synthetic corpus)")
if failures:
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
