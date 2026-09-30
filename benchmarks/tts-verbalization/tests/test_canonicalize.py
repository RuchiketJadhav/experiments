"""Golden test for the canonicalizer.

This is the test that protects the whole metric. If it passes, a correct TTS
is not penalized for the ASR writing digits. If it is too permissive, every
number matches everything and the benchmark measures nothing -- so the
NON-matches below matter as much as the matches.

Run: python tests/test_canonicalize.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from canonicalize import (_parse_number_run, best_wer,  # noqa: E402
                          canonical_variants, has_target, matches,
                          target_match, target_text)

failures: list[str] = []
checks = 0


def check(cond: bool, label: str) -> None:
    global checks
    checks += 1
    if not cond:
        failures.append(label)


# --------------------------------------------------------------------------
# 1. Number-run composition
# --------------------------------------------------------------------------
PARSE_CASES = [
    (["one", "two", "three"],                  [(1, False), (2, False), (3, False)]),
    (["one", "hundred", "twenty", "three"],    [(123, False)]),
    (["twenty", "twenty", "three"],            [(20, False), (23, False)]),
    (["nineteen", "ninety", "five"],           [(19, False), (95, False)]),
    (["two", "thousand", "twenty", "four"],    [(2024, False)]),
    (["fifteen", "forty", "five"],             [(15, False), (45, False)]),
    (["ten", "thirty"],                        [(10, False), (30, False)]),
    (["twentieth"],                            [(20, True)]),
    (["twenty", "first"],                      [(21, True)]),
    (["third", "nineteen", "ninety", "five"],  [(3, True), (19, False), (95, False)]),
    (["ninety", "nine"],                       [(99, False)]),
    # Consecutive zeros must not collapse: an ISBN ending "...one zero zero"
    # is 3 digits, not 2. A current of 0 is a real value, not an empty group.
    (["one", "zero", "zero"],                  [(1, False), (0, False), (0, False)]),
    (["zero", "zero"],                         [(0, False), (0, False)]),
    (["four", "zero", "zero", "six", "zero"],  [(4, False), (0, False), (0, False),
                                                (6, False), (0, False)]),
]
for words, expected in PARSE_CASES:
    got = _parse_number_run(words)
    check(got == expected, f"parse {' '.join(words)!r}: expected {expected}, got {got}")

# --------------------------------------------------------------------------
# 2. The crux: a normalizing ASR writes digits, the reference is words.
#    Every pair below is a CORRECT TTS rendering and must match.
# --------------------------------------------------------------------------
MATCH_CASES = [
    # (ASR hypothesis as Whisper would write it, PolyNorm spoken reference)
    ("The event is on May 20th, 2023.",
     "The event is on May twentieth twenty twenty three."),
    ("They married on July 3rd, 1995.",
     "They married on July third nineteen ninety five."),
    ("The deadline is December 31st, 2024.",
     "The deadline is December thirty first twenty twenty four."),
    ("The price is $10.99.",
     "The price is ten dollars and ninety nine cents."),
    ("She paid €25 for the book.",
     "She paid twenty five euros for the book."),
    ("Let's meet at 10:30 tomorrow.",
     "Let's meet at ten thirty tomorrow."),
    ("The train departs at 15:45.",
     "The train departs at fifteen forty five."),
    ("The address is 123 Main Street, USA.",
     "The address is one two three Main Street, U S A."),
    ("The office is at 456 Elm Avenue, Suite 200.",
     "The office is at four five six Elm Avenue, Suite two hundred."),
    ("Pi is roughly 3.14.",
     "Pi is roughly three point one four."),
    ("See Chapter IV.",
     "See Chapter four."),
    # A non-normalizing CTC ASR writes the words already -- must also match.
    ("THE EVENT IS ON MAY TWENTIETH TWENTY TWENTY THREE",
     "The event is on May twentieth twenty twenty three."),
    # Symbols the ASR writes back but the reference spells out.
    ("Visit example.com for details.",
     "Visit example dot com for details."),
    ("Email me at john@example.com.",
     "Email me at john at example dot com."),
    ("Use #AI in your post.",
     "Use hashtag A I in your post."),
    ("The result is 5 + 3 = 8.",
     "The result is five plus three equals eight."),
    ("Water is H2O.",
     "Water is H two O."),
    ("Add 1/2 cup of sugar.",
     "Add one half cup of sugar."),
    ("Add 3/4 cup of flour.",
     "Add three quarters cup of flour."),
]
for hyp, ref in MATCH_CASES:
    check(matches(hyp, ref), f"should match:\n    hyp={hyp!r}\n    ref={ref!r}")

# --------------------------------------------------------------------------
# 3. Guard against a canonicalizer that matches everything.
#    These are WRONG renderings and must NOT match.
# --------------------------------------------------------------------------
NON_MATCH_CASES = [
    # wrong value
    ("The event is on May 21st, 2023.",
     "The event is on May twentieth twenty twenty three."),
    # digits read out individually instead of as a year
    ("The event is on May 20th, 2 0 2 3.",
     "The event is on May twentieth twenty twenty three."),
    # cardinal where an ordinal was required
    ("The event is on May 20, 2023.",
     "The event is on May twentieth twenty twenty three."),
    # currency: cents dropped
    ("The price is $10.",
     "The price is ten dollars and ninety nine cents."),
    # currency: wrong unit
    ("The price is 10 pounds 99 pence.",
     "The price is ten dollars and ninety nine cents."),
    # time misread
    ("The train departs at 15:40.",
     "The train departs at fifteen forty five."),
    # wrong word entirely
    ("The train departs at fifteen forty five sharp.",
     "The train arrives at fifteen forty five."),
    # wrong domain
    ("Visit example.org for details.",
     "Visit example dot com for details."),
    # wrong fraction
    ("Add 1/3 cup of sugar.",
     "Add one half cup of sugar."),
    # wrong chemical formula
    ("Water is H2O2.",
     "Water is H two O."),
]
for hyp, ref in NON_MATCH_CASES:
    check(not matches(hyp, ref), f"should NOT match:\n    hyp={hyp!r}\n    ref={ref!r}")

# --------------------------------------------------------------------------
# 4. Segmentation offers the alternate readings, and stays bounded.
# --------------------------------------------------------------------------
v = canonical_variants("one two three")
check(("123",) in v, f"'one two three' should offer ('123',); got {sorted(v)}")
check(("1", "2", "3") in v, f"'one two three' should offer ('1','2','3'); got {sorted(v)}")
check(len(canonical_variants("one two three four five six seven eight nine ten")) <= 64,
      "variant explosion is not capped")

# 5. best_wer is 0 on a match and > 0 on a miss.
check(best_wer("The event is on May 20th, 2023.",
               "The event is on May twentieth twenty twenty three.") == 0.0,
      "best_wer should be 0.0 on a correct rendering")
check(best_wer("The event is on May 21st, 2023.",
               "The event is on May twentieth twenty twenty three.") > 0.0,
      "best_wer should be > 0 on a wrong rendering")

# --------------------------------------------------------------------------
# 6. Target-span scoring. Found by SURFACE diff, scored canonically -- so ASR
#    noise in the carrier sentence cannot fail a correct verbalization.
# --------------------------------------------------------------------------
TARGET_TEXT_CASES = [
    ("Let's meet at 10:30 tomorrow.", "Let's meet at ten thirty tomorrow.",
     "ten thirty"),
    ("The price is $10.99.", "The price is ten dollars and ninety nine cents.",
     "ten dollars and ninety nine cents"),
    ("There are 25 students in the class.",
     "There are twenty five students in the class.", "twenty five"),
]
for written, spoken, expected in TARGET_TEXT_CASES:
    got = target_text(written, spoken)
    check(got == expected, f"target_text({written!r}) -> {got!r}, wanted {expected!r}")

# The real failure that motivated this: wav2vec2 wrote "DEAD LINE" for
# "deadline", failing a PERFECT verbalization under whole-sentence scoring.
_W = "The deadline is 2024-12-31."
_S = "The deadline is December thirty first twenty twenty four."
_HYP_GOOD = "THE DEAD LINE IS DECEMBER THIRTY FIRST TWENTY TWENTY FOUR"
_HYP_BAD = "THE DEAD LINE IS TWENTY TWENTY FORT WELVE THIRTY ONE"
check(target_match(_HYP_GOOD, _W, _S, "Date"),
      "carrier-only ASR error must not fail a correct verbalization")
check(not matches(_HYP_GOOD, _S, "Date"),
      "sentence match is expected to fail here -- that is why target_match exists")
check(not target_match(_HYP_BAD, _W, _S, "Date"),
      "a genuinely garbled date must still fail")

# Real Kokoro failures from the smoke run must score as misses.
check(not target_match("THE EVENT IS ON FIVE TWENTY TWENTY TWENTY THREE",
                       "The event is on 05/20/2023.",
                       "The event is on May twentieth twenty twenty three.", "Date"),
      "reading the month '05' as 'five' must be a miss")
check(not target_match("HER BIRTHDAY IS OC TWELVE", "Her birthday is Oct 12.",
                       "Her birthday is October twelfth.", "Date"),
      "'OC TWELVE' for 'October twelfth' must be a miss")
check(target_match("THE EVENT IS ON MAY TWENTIETH TWENTY TWENTY THREE",
                   "The event is on 05/20/2023.",
                   "The event is on May twentieth twenty twenty three.", "Date"),
      "a correct date rendering must match")

# An item whose written and spoken forms are identical has nothing to test.
check(not has_target("Homo sapiens is us.", "Homo sapiens is us."),
      "identical written/spoken must report no target")
check(has_target("Let's meet at 10:30 tomorrow.",
                 "Let's meet at ten thirty tomorrow."),
      "a normalized item must report a target")

# --------------------------------------------------------------------------
print(f"{checks} checks run")
if failures:
    print(f"\n{len(failures)} FAILED:\n")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("all passed")
