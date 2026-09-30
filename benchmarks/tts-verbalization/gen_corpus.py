"""Generate a synthetic text-normalization corpus we own outright.

Why this exists
---------------
Every number this project has produced sits on Apple's PolyNorm-Bench, which
is CC BY-NC-ND 4.0: the results may be published, the corpus may not. That
blocks the actual goal -- shipping a benchmark somebody else can run and
check. A corpus generated here removes the constraint: data, code and results
publish together.

How it works
------------
Generation is DETERMINISTIC and PROGRAMMATIC, not LLM-written. Every category
has a value generator that emits the written form and its correct spoken form
from the same structured value, so the reference reading is correct by
construction. There is nothing to verify with a second model and nothing for a
human to adjudicate, because no model ever guessed at the answer. A seed makes
the whole corpus reproducible from this file plus data/spec/<locale>.yaml.

The language seam is the YAML: carrier sentences and counts live there. The
value generators here encode English number grammar and English reading
conventions, so a second locale is a new spec file plus a new generator table,
not a rewrite of the pipeline.

Self-checks (see verify())
--------------------------
Each candidate must pass three tests against canonicalize.py before it enters
the corpus:

  1. has_target  -- written and spoken must not canonicalize to the same
     multiset. An item with no target cannot tell a correct rendering from a
     wrong one.
  2. word-form round trip -- target_match(spoken, ...) must hold. If the
     reference reading itself does not score as correct, the item is unwinnable.
  3. written-form round trip -- target_match(written, ...) must hold. A
     recognizer that hears a correct rendering usually writes it back in the
     WRITTEN form ("ten thirty" -> "10:30"); if the scorer rejects that, every
     correct model is marked wrong.

A fourth test runs as a negative control: a deliberately corrupted reading
must NOT match, which proves the item detects an error at all.

Items failing any test are written to rejects.jsonl WITH THE REASON rather
than silently dropped. That file is the audit trail for the one real hazard
here: filtering on the current canonicalizer selects for transformations it
already understands, so a blind spot would quietly delete a category with
nothing in the output saying so. The rejects file says so.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from canonicalize import has_target, target_match  # noqa: E402

SPEC_DIR = ROOT / "data" / "spec"
OUT_DIR = ROOT / "data" / "synthetic"

# Upper bound on how fast anything speaks, used to keep generated items clear
# of screen.py's implausibly_long rule (0.60 s of audio per character of
# WRITTEN text). A spoken form of N characters takes roughly N/13 seconds; the
# factor of 2 is headroom for a slow voice.
CHARS_PER_SECOND = 13.0
DURATION_SAFETY = 2.0

POOL_FACTOR = 12          # candidates generated per accepted item


# --------------------------------------------------------------------------
# English number words.
# --------------------------------------------------------------------------

UNITS = ["zero", "one", "two", "three", "four", "five", "six", "seven",
         "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
         "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty", 6: "sixty",
        7: "seventy", 8: "eighty", 9: "ninety"}
SCALES = [(10 ** 9, "billion"), (10 ** 6, "million"), (1000, "thousand")]

ORDINAL_IRREGULAR = {
    "one": "first", "two": "second", "three": "third", "five": "fifth",
    "eight": "eighth", "nine": "ninth", "twelve": "twelfth",
    "twenty": "twentieth", "thirty": "thirtieth", "forty": "fortieth",
    "fifty": "fiftieth", "sixty": "sixtieth", "seventy": "seventieth",
    "eighty": "eightieth", "ninety": "ninetieth",
}


def cardinal(n: int) -> str:
    """0 .. 999,999,999,999 in words, American style (no 'and')."""
    if n < 0:
        return "minus " + cardinal(-n)
    if n < 20:
        return UNITS[n]
    if n < 100:
        t, r = divmod(n, 10)
        return TENS[t] + (" " + UNITS[r] if r else "")
    if n < 1000:
        h, r = divmod(n, 100)
        return UNITS[h] + " hundred" + (" " + cardinal(r) if r else "")
    for value, name in SCALES:
        if n >= value:
            q, r = divmod(n, value)
            return cardinal(q) + " " + name + (" " + cardinal(r) if r else "")
    raise ValueError(f"out of range: {n}")


def ordinal_words(n: int) -> str:
    words = cardinal(n).split()
    last = words[-1]
    if last in ORDINAL_IRREGULAR:
        words[-1] = ORDINAL_IRREGULAR[last]
    elif last.endswith("y"):
        words[-1] = last[:-1] + "ieth"
    else:
        words[-1] = last + "th"
    return " ".join(words)


def ordinal_digits(n: int) -> str:
    """47 -> '47th'. The written form of an ordinal."""
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def digit_words(s: str) -> str:
    """'4567' -> 'four five six seven'. Non-digits are dropped."""
    return " ".join(UNITS[int(c)] for c in s if c.isdigit())


def letter_words(s: str) -> str:
    """'FBI' -> 'F B I'. Non-letters are dropped."""
    return " ".join(c for c in s if c.isalpha())


def year_words(y: int) -> str:
    hi, lo = divmod(y, 100)
    if 2000 <= y <= 2009:
        return "two thousand" + (" " + cardinal(lo) if lo else "")
    if lo == 0:
        return cardinal(hi) + " hundred"
    if lo < 10:
        return cardinal(hi) + " oh " + cardinal(lo)
    return cardinal(hi) + " " + cardinal(lo)


def decimal_words(whole: int, frac: str) -> str:
    """3, '14159' -> 'three point one four one five nine'."""
    return cardinal(whole) + " point " + digit_words(frac)


def alnum_words(s: str) -> str:
    """'ABC-1234' -> 'A B C one two three four'; separators are silent."""
    out = []
    for c in s:
        if c.isdigit():
            out.append(UNITS[int(c)])
        elif c.isalpha():
            out.append(c)
    return " ".join(out)


# --------------------------------------------------------------------------
# Vocabulary used by the value generators.
# --------------------------------------------------------------------------

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug",
              "Sep", "Oct", "Nov", "Dec"]
DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

FRACTION_NAMES = {2: ("half", "halves"), 3: ("third", "thirds"),
                  4: ("quarter", "quarters"), 5: ("fifth", "fifths"),
                  6: ("sixth", "sixths"), 7: ("seventh", "sevenths"),
                  8: ("eighth", "eighths"), 9: ("ninth", "ninths"),
                  10: ("tenth", "tenths")}

# Only units canonicalize.py can read back from an abbreviation, plus units a
# recognizer writes as words anyway. A unit outside this set (mph, GB) would
# score a correct rendering as wrong whenever the recognizer writes the
# abbreviation, which measures the scorer rather than the voice.
ABBREV_UNITS = [("kg", "kilograms"), ("mg", "milligrams"), ("km", "kilometers"),
                ("cm", "centimeters"), ("mm", "millimeters"),
                ("ml", "milliliters"), ("lb", "pounds"), ("oz", "ounces"),
                ("ft", "feet"), ("hr", "hours"), ("sec", "seconds")]
WORD_UNITS = ["miles", "liters", "meters", "grams", "inches", "yards",
              "gallons", "minutes", "degrees", "percent"]

HONORIFICS = [("Dr.", "Doctor"), ("Mr.", "Mister"), ("Mrs.", "Missus"),
              ("Ms.", "Miss"), ("Prof.", "Professor"), ("Sgt.", "Sergeant"),
              ("Capt.", "Captain"), ("Rev.", "Reverend"), ("Gov.", "Governor"),
              ("Sen.", "Senator"), ("Lt.", "Lieutenant"), ("Col.", "Colonel")]

INITIALISMS = ["FBI", "CIA", "IRS", "FDA", "EPA", "NIH", "BBC", "NHS", "IMF",
               "WTO", "WHO", "CDC", "FTC", "FCC", "SEC", "DOJ", "DHS", "NSA",
               "GAO", "USDA", "USGS", "NOAA", "OECD", "IAEA", "NTSB", "FAA",
               "TSA", "ATF", "EEOC", "USPS", "DMV", "UNDP", "ICRC", "ILO",
               "NATO", "IEEE", "ISO", "BLS", "CBO", "FEC", "OSHA", "SEC",
               "UNHCR", "NAACP", "AFL", "ACLU"]

# Single-digit subscripts only. "C6H12O6" has a genuinely ambiguous reading
# ("H one two" vs "H twelve") and an ambiguous reference is a broken test.
FORMULAS = ["H2O", "CO2", "NaCl", "H2SO4", "NH3", "CaCO3", "O2", "N2", "CH4",
            "HCl", "NaOH", "KCl", "ZnO", "MgO", "H2O2", "SO2", "NO2", "CaO",
            "FeS", "KBr", "NaF", "LiOH", "BaO", "AgCl", "CuO", "PbS", "SnO2",
            "TiO2", "SiO2", "Al2O3", "Fe2O3", "CuSO4", "MgSO4", "KMnO4",
            "AgNO3", "NaHCO3", "K2CO3", "Na2SO4", "CaCl2", "NH4Cl", "H3PO4",
            "CaSO4", "Na2CO3", "KNO3", "MnO2", "CrO3", "NiCl2", "CoCl2"]

TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "NVDA", "META", "NFLX",
           "INTC", "AMD", "ORCL", "CRM", "ADBE", "PYPL", "CSCO", "QCOM",
           "TXN", "IBM", "BA", "GE", "JPM", "BAC", "WFC", "GS", "MS", "C",
           "XOM", "CVX", "PFE", "MRK", "JNJ", "ABBV", "KO", "PEP", "WMT",
           "TGT", "COST", "HD", "LOW", "MCD", "SBUX", "NKE", "DIS", "UBER",
           "LYFT", "SNAP", "SQ", "SHOP"]

GENUS_SPECIES = [("E.", "Escherichia", "coli"), ("S.", "Staphylococcus", "aureus"),
                 ("H.", "Homo", "sapiens"), ("C.", "Clostridium", "difficile"),
                 ("M.", "Mycobacterium", "tuberculosis"), ("P.", "Pseudomonas", "aeruginosa"),
                 ("B.", "Bacillus", "subtilis"), ("L.", "Listeria", "monocytogenes"),
                 ("V.", "Vibrio", "cholerae"), ("D.", "Drosophila", "melanogaster"),
                 ("A.", "Arabidopsis", "thaliana"), ("S.", "Salmonella", "enterica")]

STREET_NAMES = ["Main", "Elm", "Oak", "Maple", "Cedar", "Pine", "Birch",
                "Walnut", "Chestnut", "Willow", "Juniper", "Sycamore",
                "Bradford", "Kingsley", "Harper", "Lowell", "Ashford",
                "Merrick", "Danforth", "Wexley"]
STREET_SUFFIX = [("St", "Street"), ("Ave", "Avenue"), ("Blvd", "Boulevard"),
                 ("Rd", "Road"), ("Ln", "Lane"), ("Ct", "Court"),
                 ("Pl", "Place"), ("Pkwy", "Parkway"), ("Sq", "Square"),
                 ("Hwy", "Highway")]
UNIT_PREFIX = [("Apt", "Apartment"), ("Ste", "Suite"), ("Unit", "Unit")]

HASHTAG_WORDS = ["travel", "tips", "throwback", "thursday", "open", "source",
                 "world", "cup", "data", "science", "spring", "clean",
                 "coffee", "break", "book", "club", "road", "trip", "night",
                 "shift", "deep", "work", "city", "guide", "home", "made",
                 "slow", "food", "winter", "sale", "green", "energy"]
MENTION_WORDS = ["alice", "bob", "carol", "dave", "erin", "frank", "grace",
                 "henry", "irene", "jamal", "kira", "liam", "maya", "noor",
                 "omar", "priya", "quinn", "rosa", "sami", "tariq"]

DOMAINS = ["example", "archive", "granite", "northfield", "harborlight",
           "quietriver", "bluestem", "ironwood", "saltmarsh", "clearwater"]
TLDS = ["com", "org", "net", "io", "dev", "co"]
PATHS = ["docs", "guide", "intro", "support", "faq", "news", "blog",
         "downloads", "reference", "start"]
EMAIL_FIRST = ["john", "sara", "miguel", "anita", "peter", "lena", "oscar",
               "ruth", "tomas", "yuki"]
EMAIL_LAST = ["doe", "patel", "kowalski", "nguyen", "oconnor", "silva",
              "muller", "haddad", "ferreira", "novak"]

NOTE_LETTERS = ["A", "B", "C", "D", "E", "F", "G"]

LEGAL_TITLES = [("Section", "Section"), ("Article", "Article"),
                ("Clause", "Clause"), ("Title", "Title"),
                ("Chapter", "Chapter"), ("Rule", "Rule"), ("Part", "Part")]

CODE_LETTERS = "ABCDEFGHJKLMNPRSTVWXYZ"     # no I, O, Q, U -- plate conventions


def roman(n: int) -> str:
    table = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"),
             (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"),
             (5, "V"), (4, "IV"), (1, "I")]
    out = []
    for value, sym in table:
        while n >= value:
            out.append(sym)
            n -= value
    return "".join(out)


# --------------------------------------------------------------------------
# Value generators. Each returns (written_fragment, spoken_fragment).
# --------------------------------------------------------------------------

def g_date(r: random.Random) -> tuple[str, str]:
    m = r.randrange(12)
    d = r.randint(1, DAYS_IN_MONTH[m])
    y = r.randint(1950, 2035)
    style = r.randrange(4)
    if style == 0:
        w = f"{m + 1:02d}/{d:02d}/{y}"
        s = f"{MONTHS[m]} {ordinal_words(d)} {year_words(y)}"
    elif style == 1:
        w = f"{y}-{m + 1:02d}-{d:02d}"
        s = f"{MONTHS[m]} {ordinal_words(d)} {year_words(y)}"
    elif style == 2:
        w = f"{MONTH_ABBR[m]} {d}"
        s = f"{MONTHS[m]} {ordinal_words(d)}"
    else:
        w = f"{ordinal_digits(d)} {MONTHS[m]} {y}"
        s = f"{MONTHS[m]} {ordinal_words(d)} {year_words(y)}"
    return w, s


def g_time(r: random.Random) -> tuple[str, str]:
    style = r.randrange(3)
    if style == 0:                      # 12-hour clock
        h, mi = r.randint(1, 12), r.randint(1, 59)
    elif style == 1:                    # 24-hour clock
        h, mi = r.randint(13, 23), r.randint(1, 59)
    else:
        h = r.randint(1, 12)
        ap = r.choice(["AM", "PM"])
        return f"{h} {ap}", f"{cardinal(h)} {ap[0]} {ap[1]}"
    minute = cardinal(mi) if mi >= 10 else "oh " + cardinal(mi)
    return f"{h}:{mi:02d}", f"{cardinal(h)} {minute}"


def g_cardinal(r: random.Random) -> tuple[str, str]:
    digits = r.randint(4, 9)
    n = r.randrange(10 ** (digits - 1), 10 ** digits)
    return f"{n:,}", cardinal(n)


def g_ordinal(r: random.Random) -> tuple[str, str]:
    n = r.choice([r.randint(2, 99), r.randint(100, 999)])
    return ordinal_digits(n), ordinal_words(n)


def g_decimal(r: random.Random) -> tuple[str, str]:
    whole = r.randrange(0, 1000)
    frac = "".join(str(r.randrange(10)) for _ in range(r.randint(2, 6)))
    return f"{whole}.{frac}", decimal_words(whole, frac)


def g_fraction(r: random.Random) -> tuple[str, str]:
    den = r.randint(2, 10)
    num = r.randint(1, den - 1)
    singular, plural = FRACTION_NAMES[den]
    name = singular if num == 1 else plural
    return f"{num}/{den}", f"{cardinal(num)} {name}"


def g_roman(r: random.Random) -> tuple[str, str]:
    n = r.randint(2, 400)
    return roman(n), cardinal(n)


def g_phone(r: random.Random) -> tuple[str, str]:
    area = r.choice(["212", "415", "312", "617", "305", "206", "503", "800",
                     "702", "404", "713", "919"])
    mid = f"{r.randrange(200, 999)}"
    last = f"{r.randrange(0, 10000):04d}"
    style = r.randrange(3)
    if style == 0:
        w = f"{area}-{mid}-{last}"
    elif style == 1:
        w = f"({area}) {mid}-{last}"
    else:
        w = f"1-{area}-{mid}-{last}"
    return w, digit_words(w)


def g_currency(r: random.Random) -> tuple[str, str]:
    style = r.randrange(3)
    if style == 0:
        whole, cents = r.randrange(1, 1000), r.randrange(1, 100)
    elif style == 1:
        whole, cents = r.randrange(1000, 10_000_000), r.randrange(1, 100)
    else:
        whole, cents = r.randrange(5, 5000), 0
    w = f"${whole:,}" + (f".{cents:02d}" if cents else "")
    s = cardinal(whole) + " dollars"
    if cents:
        s += " and " + cardinal(cents) + " cents"
    return w, s


def g_unit(r: random.Random) -> tuple[str, str]:
    if r.random() < 0.6:
        abbr, word = r.choice(ABBREV_UNITS)
    else:
        abbr = word = r.choice(WORD_UNITS)
    n = r.choice([r.randint(2, 99), r.randrange(100, 10000)])
    return f"{n} {abbr}", f"{cardinal(n)} {word}"


def g_score(r: random.Random) -> tuple[str, str]:
    a = r.randint(0, 120)
    b = r.randint(0, 120)
    return f"{a}-{b}", f"{cardinal(a)} to {cardinal(b)}"


def g_math(r: random.Random) -> tuple[str, str]:
    op = r.choice(["+", "-", "%"])
    if op == "+":
        a, b = r.randint(2, 500), r.randint(2, 500)
        return f"{a} + {b} = {a + b}", f"{cardinal(a)} plus {cardinal(b)} equals {cardinal(a + b)}"
    if op == "-":
        a, b = r.randint(50, 900), r.randint(2, 49)
        return f"{a} - {b} = {a - b}", f"{cardinal(a)} minus {cardinal(b)} equals {cardinal(a - b)}"
    pct, total = r.choice([10, 20, 25, 50, 75]), r.randrange(100, 1000, 20)
    return (f"{pct}% of {total} = {pct * total // 100}",
            f"{cardinal(pct)} percent of {cardinal(total)} equals {cardinal(pct * total // 100)}")


def g_url(r: random.Random) -> tuple[str, str]:
    style = r.randrange(3)
    dom, tld = r.choice(DOMAINS), r.choice(TLDS)
    if style == 0:
        scheme = r.choice(["http", "https"])
        w = f"{scheme}://www.{dom}.{tld}"
        s = f"{letter_words(scheme)} colon slash slash w w w dot {dom} dot {tld}"
    elif style == 1:
        first, last = r.choice(EMAIL_FIRST), r.choice(EMAIL_LAST)
        w = f"{first}.{last}@{dom}.{tld}"
        s = f"{first} dot {last} at {dom} dot {tld}"
    else:
        path = r.choice(PATHS)
        w = f"{dom}.{tld}/{path}"
        s = f"{dom} dot {tld} slash {path}"
    return w, s


def g_hashtag(r: random.Random) -> tuple[str, str]:
    if r.random() < 0.6:
        a, b = r.sample(HASHTAG_WORDS, 2)
        return "#" + a.capitalize() + b.capitalize(), f"hashtag {a} {b}"
    a, b = r.sample(MENTION_WORDS, 2)
    return f"@{a}_{b}", f"at {a} underscore {b}"


def g_abbrev(r: random.Random) -> tuple[str, str]:
    return r.choice(HONORIFICS)


def g_initialism(r: random.Random) -> tuple[str, str]:
    s = r.choice(INITIALISMS)
    return s, letter_words(s)


def g_isbn(r: random.Random) -> tuple[str, str]:
    if r.random() < 0.6:
        body = f"978-{r.randrange(10)}-{r.randrange(10, 100)}-{r.randrange(100000, 1000000)}-{r.randrange(10)}"
        label_w, label_s = "ISBN", "I S B N"
    else:
        body = f"{r.randrange(10)}-{r.randrange(100, 1000)}-{r.randrange(10000, 100000)}-{r.randrange(10)}"
        label_w, label_s = "ISBN-10", "I S B N ten"
    return f"{label_w} {body}", f"{label_s} {digit_words(body)}"


def g_formula(r: random.Random) -> tuple[str, str]:
    f = r.choice(FORMULAS)
    return f, alnum_words(f)


def g_legal(r: random.Random) -> tuple[str, str]:
    style = r.randrange(3)
    if style == 0:
        title, spoken_title = r.choice(LEGAL_TITLES)
        a, b = r.randint(1, 40), r.randint(1, 20)
        return f"{title} {a}.{b}", f"{spoken_title} {cardinal(a)} point {cardinal(b)}"
    if style == 1:
        # One non-standard span per item. "Article XXX, Clause 12" has two,
        # and target_text joins the replaced spans without the words between
        # them, so the reference target became the single number "thirty
        # twelve" -> 42 and the item could never match itself.
        n = r.randint(2, 30)
        return f"Article {roman(n)}", f"Article {cardinal(n)}"
    title = r.randint(5, 50)
    sec = r.randint(100, 9999)
    return (f"{title} U.S.C. {sec}",
            f"{cardinal(title)} U S C {cardinal(sec)}")


def g_product(r: random.Random) -> tuple[str, str]:
    style = r.randrange(3)
    letters = "".join(r.choice(CODE_LETTERS) for _ in range(r.randint(2, 3)))
    if style == 0:
        digits = f"{r.randrange(1000, 10000)}"
        w = f"{letters}-{digits}"
    elif style == 1:
        digits = f"{r.randrange(100000, 1000000)}"
        w = f"{letters}{digits}"
    else:
        digits = f"{r.randrange(10, 100)}"
        w = f"{letters}-{digits}-{r.choice(CODE_LETTERS)}{r.randrange(10, 100)}"
    return w, alnum_words(w)


def g_coords(r: random.Random) -> tuple[str, str]:
    lat_w, lat_f = r.randrange(0, 90), f"{r.randrange(0, 10000):04d}"
    lon_w, lon_f = r.randrange(0, 180), f"{r.randrange(0, 10000):04d}"
    ns = r.choice([("N", "north"), ("S", "south")])
    ew = r.choice([("E", "east"), ("W", "west")])
    if r.random() < 0.3:
        return (f"{lat_w}.{lat_f}° {ns[0]}",
                f"{decimal_words(lat_w, lat_f)} degrees {ns[1]}")
    return (f"{lat_w}.{lat_f}° {ns[0]}, {lon_w}.{lon_f}° {ew[0]}",
            f"{decimal_words(lat_w, lat_f)} degrees {ns[1]}, "
            f"{decimal_words(lon_w, lon_f)} degrees {ew[1]}")


def g_version(r: random.Random) -> tuple[str, str]:
    parts = [r.randrange(0, 30) for _ in range(r.randint(2, 3))]
    body_w = ".".join(str(p) for p in parts)
    body_s = " point ".join(cardinal(p) for p in parts)
    if r.random() < 0.5:
        return f"v{body_w}", f"version {body_s}"
    return f"version {body_w}", f"version {body_s}"


def g_serial(r: random.Random) -> tuple[str, str]:
    style = r.randrange(3)
    if style == 0:                                   # license plate
        w = ("".join(r.choice(CODE_LETTERS) for _ in range(3))
             + "-" + f"{r.randrange(1000, 10000)}")
    elif style == 1:
        w = (f"{r.randrange(1, 10)}"
             + "".join(r.choice(CODE_LETTERS) for _ in range(2))
             + f"{r.randrange(100, 1000)}"
             + "".join(r.choice(CODE_LETTERS) for _ in range(2)))
    else:
        w = ("SN-" + "".join(r.choice(CODE_LETTERS) for _ in range(2))
             + f"{r.randrange(10000, 100000)}")
    return w, alnum_words(w)


def g_music(r: random.Random) -> tuple[str, str]:
    style = r.randrange(4)
    note = r.choice(NOTE_LETTERS)
    if style == 0:
        acc, acc_s = r.choice([("♯", "sharp"), ("♭", "flat")])
        mode = r.choice(["minor", "major"])
        return f"{note}{acc} {mode}", f"{note} {acc_s} {mode}"
    if style == 1:
        n = r.choice([6, 7, 9])
        return f"{note}{n}", f"{note} {cardinal(n)}"
    if style == 2:
        num, den = r.choice([(4, 4), (3, 4), (6, 8), (2, 4), (12, 8), (5, 4)])
        return (f"{num}/{den} time",
                f"{cardinal(num)} {cardinal(den)} time")
    acc, acc_s = r.choice([("♯", "sharp"), ("♭", "flat")])
    return f"{note}{acc}", f"{note} {acc_s}"


def g_ticker(r: random.Random) -> tuple[str, str]:
    t = r.choice(TICKERS)
    return t, letter_words(t)


def g_bio(r: random.Random) -> tuple[str, str]:
    if r.random() < 0.6:
        initial, genus, species = r.choice(GENUS_SPECIES)
        return f"{initial} {species}", f"{genus} {species}"
    letter1 = r.choice("HNPB")
    letter2 = r.choice("NSH")
    a, b = r.randint(1, 9), r.randint(1, 9)
    w = f"{letter1}{a}{letter2}{b}"
    return w, alnum_words(w)


def g_address(r: random.Random) -> tuple[str, str]:
    num = r.randrange(100, 10000)
    street = r.choice(STREET_NAMES)
    suf_w, suf_s = r.choice(STREET_SUFFIX)
    base_w = f"{num} {street} {suf_w}"
    base_s = f"{digit_words(str(num))} {street} {suf_s}"
    if r.random() < 0.5:
        pre_w, pre_s = r.choice(UNIT_PREFIX)
        n = r.randrange(1, 500)
        return (f"{base_w}, {pre_w} {n}", f"{base_s}, {pre_s} {cardinal(n)}")
    return base_w, base_s


GENERATORS = {
    "Date": g_date,
    "Time": g_time,
    "Cardinal": g_cardinal,
    "Ordinal": g_ordinal,
    "Decimal": g_decimal,
    "Fractions": g_fraction,
    "Roman Numeral": g_roman,
    "Phone Number": g_phone,
    "Currency": g_currency,
    "Unit": g_unit,
    "Sports score": g_score,
    "Mathematical Expression": g_math,
    "URL or Email": g_url,
    "Hashtag or Mention": g_hashtag,
    "Abbreviation": g_abbrev,
    "Initialism or Acronym": g_initialism,
    "ISBN": g_isbn,
    "Chemical Formula": g_formula,
    "Legal Reference": g_legal,
    "Vehicle or Product Code": g_product,
    "Geographic Coordinates": g_coords,
    "Version Numbers": g_version,
    "License Plate or Serial Numbers": g_serial,
    "Musical Notation": g_music,
    "Stock Ticker": g_ticker,
    "Biological Classification": g_bio,
    "Address": g_address,
}


# --------------------------------------------------------------------------
# Verification.
# --------------------------------------------------------------------------

def corrupt(spoken_fragment: str) -> str | None:
    """Change one number word, so the result is a WRONG reading of the item.

    Used as a negative control: an item that still scores as correct after
    this has a target the scorer cannot actually discriminate on.
    """
    swaps = {"one": "nine", "two": "seven", "three": "eight", "four": "one",
             "five": "two", "six": "three", "seven": "four", "eight": "five",
             "nine": "six", "zero": "four", "ten": "twelve",
             "twenty": "fifty", "thirty": "seventy", "hundred": "thousand",
             "first": "ninth", "second": "seventh", "third": "eighth"}
    toks = spoken_fragment.split()
    for i, t in enumerate(toks):
        key = t.strip(".,").lower()
        if key in swaps:
            toks[i] = swaps[key]
            return " ".join(toks)
    return None


def verify(written: str, spoken: str, frag_w: str, frag_s: str,
           category: str) -> tuple[str, bool]:
    """(reason, writeback_scores).

    `reason` is '' when the item is sound. `writeback_scores` records whether
    a recognizer that echoes the WRITTEN form verbatim would still score the
    item correct -- true for "10:30", false for "2009-02-26", which no
    recognizer would ever emit for spoken English anyway. It is a recorded
    property, not a gate: gating on it would delete every hard Date and
    Version item and leave a corpus shaped by what the canonicalizer already
    knows how to undo.
    """
    if not has_target(written, spoken, category):
        return "no_target", False
    if not target_match(spoken, written, spoken, category):
        return "word_form_rejected", False
    # The negative control corrupts the FRAGMENT, never the carrier: swapping
    # a number in "in the second room" proves nothing about the item's target
    # and rejected a third of the Abbreviation and Ticker candidates.
    bad_frag = corrupt(frag_s)
    if bad_frag is not None:
        bad = spoken.replace(frag_s, bad_frag, 1)
        if bad != spoken and target_match(bad, written, spoken, category):
            return "corruption_accepted", False
    est = len(spoken) / CHARS_PER_SECOND * DURATION_SAFETY
    if est > 0.60 * len(written):
        return f"screen_risk:{est:.1f}s_vs_{0.60 * len(written):.1f}s", False
    return "", target_match(written, written, spoken, category)


# --------------------------------------------------------------------------
# Assembly.
# --------------------------------------------------------------------------

def slug(category: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in category.lower()).strip("-")


def generate(spec: dict, seed: int, count_override: int | None = None):
    """Yield (row, reject) pairs; exactly one of the two is None."""
    default_count = int(spec.get("default_count", 40))
    locale = spec.get("locale", "en-US")
    for category, entry in spec["categories"].items():
        if category not in GENERATORS:
            raise KeyError(f"no value generator for category {category!r}")
        gen = GENERATORS[category]
        carriers = entry["carriers"]
        want = count_override or int(entry.get("count", default_count))
        # Seed per category so adding a category never reshuffles the others.
        r = random.Random(f"{seed}:{locale}:{category}")
        seen: set[str] = set()
        accepted = 0
        attempts = 0
        cat_slug = slug(category)
        while accepted < want and attempts < want * POOL_FACTOR:
            attempts += 1
            frag_w, frag_s = gen(r)
            carrier = carriers[attempts % len(carriers)]
            written = carrier.replace("{x}", frag_w)
            spoken = carrier.replace("{x}", frag_s)
            if written in seen:
                continue
            seen.add(written)
            reason, writeback = verify(written, spoken, frag_w, frag_s, category)
            rec = {"category": category, "written": written, "spoken": spoken,
                   "fragment_written": frag_w, "fragment_spoken": frag_s}
            if reason:
                yield None, dict(rec, reason=reason)
                continue
            accepted += 1
            yield dict(rec, id=f"syn-{locale.split('-')[0]}-{cat_slug}-{accepted:03d}",
                       writeback_scores=writeback), None
        if accepted < want:
            print(f"[warn] {category}: only {accepted}/{want} items passed "
                  f"verification in {attempts} attempts", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--locale", default="en-US")
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--count", type=int, default=None,
                    help="override the per-category count in the spec")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--rejects", type=Path, default=None)
    args = ap.parse_args()

    spec_path = SPEC_DIR / f"{args.locale.split('-')[0]}.yaml"
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))

    out = args.out or OUT_DIR / f"{args.locale}.jsonl"
    rej = args.rejects or OUT_DIR / f"{args.locale}.rejects.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    rows, rejects = [], []
    for row, reject in generate(spec, args.seed, args.count):
        (rows if row else rejects).append(row or reject)

    # Write atomically: a half-written corpus that still parses is worse than
    # no corpus, because nothing downstream would notice.
    def dump(path: Path, records: list[dict]) -> None:
        tmp = path.with_suffix(path.suffix + ".part")
        # newline="" pins LF on every platform. Without it Python's default
        # translation writes CRLF on Windows and LF elsewhere, so the same
        # seed produced different BYTES on different machines and the corpus
        # stopped being reproducible across operating systems.
        with tmp.open("w", encoding="utf-8", newline="") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        tmp.replace(path)

    dump(out, rows)
    dump(rej, rejects)

    from collections import Counter
    by_cat = Counter(r["category"] for r in rows)
    by_reason = Counter(r["reason"].split(":")[0] for r in rejects)
    no_writeback = Counter(r["category"] for r in rows if not r["writeback_scores"])
    print(f"wrote {len(rows)} items to {out}")
    print(f"     {len(rejects)} rejects to {rej}")
    print(f"     seed={args.seed} locale={args.locale} "
          f"spec_version={spec.get('version')}")
    short = {c: n for c, n in sorted(by_cat.items()) if n < (args.count or spec.get('default_count', 40))}
    if short:
        print("under target:", short)
    if by_reason:
        print("reject reasons:", dict(by_reason))
    if no_writeback:
        print("word-form only (a recognizer echoing the written form would not "
              "score these; it never does for these forms):",
              dict(sorted(no_writeback.items())))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
