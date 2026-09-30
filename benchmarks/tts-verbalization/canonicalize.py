"""Canonicalize written and spoken text into one comparable token space.

WHY THIS EXISTS
---------------
The benchmark feeds a TTS the *written* form and checks whether it *says* the
*spoken* form. But a normalizing ASR (Whisper) transcribes speech back into
written form: audio saying "May twentieth twenty twenty three" comes back as
"May 20th, 2023". Compared word-by-word against the spoken reference, a
perfectly correct TTS is penalized on every item -- you would be measuring the
ASR's output conventions, not the TTS.

So both sides are mapped into a canonical space where number words and digits
collapse to the same tokens.

A second problem this solves: PolyNorm supplies exactly one reference
verbalization, but several are correct ("one two three" vs "one hundred twenty
three" for 123). Each numeric run therefore expands to a SET of acceptable
canonical forms, and a hypothesis matches if any variant lines up.

Ordinality is preserved ("20#o") because Cardinal and Ordinal are distinct
categories in the corpus and collapsing them would hide real errors.
"""

from __future__ import annotations

import itertools
import re
from typing import Iterable

# Cartesian-product guard. Truncation causes silent FALSE NEGATIVES (the
# matching variant may be past the cut), so keep this well above what these
# short sentences generate rather than tuning it down for speed.
MAX_VARIANTS = 4096

UNITS = {
    "zero": 0, "oh": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fourty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
SCALES = {"hundred": 100, "thousand": 1000, "million": 10**6, "billion": 10**9}

ORDINAL_UNITS = {
    "zeroth": 0, "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14,
    "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
    "nineteenth": 19,
}
ORDINAL_TENS = {
    "twentieth": 20, "thirtieth": 30, "fortieth": 40, "fiftieth": 50,
    "sixtieth": 60, "seventieth": 70, "eightieth": 80, "ninetieth": 90,
}
ORDINAL_SCALES = {"hundredth": 100, "thousandth": 1000, "millionth": 10**6}

NUMBER_WORDS = (
    set(UNITS) | set(TENS) | set(SCALES)
    | set(ORDINAL_UNITS) | set(ORDINAL_TENS) | set(ORDINAL_SCALES)
    | {"point"}
)

# "and" joins number and currency parts ("ten dollars and ninety nine cents").
# Dropping it globally is safe on this corpus and removes a whole error class.
STOPWORDS = {"and"}

CURRENCY_SYMBOLS = {
    "$": ("dollars", "cents"),
    "€": ("euros", "cents"),
    "£": ("pounds", "pence"),
    "¥": ("yen", "sen"),
}

# Symbols an ASR writes back but the reference spells out. Ambiguous ones carry
# several readings; the variant machinery already handles alternatives, so we
# never have to guess which one the speaker used.
SYMBOL_READINGS: dict[str, tuple[str, ...]] = {
    ".": ("dot", "point", ""),     # silent in "U.S.A."
    "@": ("at",),
    "/": ("slash",),
    "#": ("hashtag", "number", "sharp"),
    "+": ("plus",),
    "=": ("equals",),
    "%": ("percent",),
    "&": ("ampersand",),
    "*": ("star", "asterisk"),
    "°": ("degrees", "degree"),
    "_": ("underscore",),
    "§": ("section",),
    "♯": ("sharp",),
    "♭": ("flat",),
    ":": ("", "colon", "to"),      # silent in "10:30", spoken in "2:1"
    # "-" is doing four jobs in this corpus: silent inside a serial or an
    # ISBN, spoken as "dash" in a case number, read "to" in a sports score
    # ("3-1" -> "three to one") and "minus" in an expression ("60 - 24").
    # Every reading is offered because the variant machinery scores the one
    # that fits; leaving "to"/"minus" out marked correct speech wrong whenever
    # the recogniser wrote the score or sum back in symbols.
    "-": ("dash", "hyphen", "", "to", "minus"),
}

# Inside an identifier -- an ISBN, a phone number, a serial -- a hyphen is
# never "to" or "minus". Offering those readings there is not just useless, it
# is harmful: "978-0-13-468599-1" has four hyphens, so five readings each is
# 625 combinations, and target_match only ever inspects the first _PAIR_CAP of
# them. The reading that actually matches gets truncated away and a correct
# ISBN scores as wrong. The identifier categories are exactly the ones that
# already set merge_digits, so that flag selects the tighter table.
IDENTIFIER_SYMBOL_READINGS: dict[str, tuple[str, ...]] = {
    "-": ("dash", "hyphen", ""),
}
SYMBOL_CHARS = "".join(re.escape(c) for c in SYMBOL_READINGS)

# Denominator words -> canonical fraction marker ("4f" == quarters).
FRACTION_WORDS = {
    "half": 2, "halves": 2, "quarter": 4, "quarters": 4,
    "third": 3, "thirds": 3, "fourth": 4, "fourths": 4, "fifth": 5, "fifths": 5,
    "sixth": 6, "sixths": 6, "seventh": 7, "sevenths": 7, "eighth": 8,
    "eighths": 8, "ninth": 9, "ninths": 9, "tenth": 10, "tenths": 10,
}

COMPASS_LETTERS = {"n": "north", "s": "south", "e": "east", "w": "west"}

ROMAN = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
ROMAN_RE = re.compile(r"^(?=[ivxlcdm]{2,}$)m*(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})$")


# Whisper writes back the written form, which means ordinary abbreviations:
# "Dr. Lee", "Lakers vs. Celtics", "250 mg". The reference says them in full.
# Both readings are offered; neither replaces the other, so a system that says
# "em gee" is still scored against what it actually said.
ABBREVIATIONS: dict[str, tuple[tuple[str, ...], ...]] = {
    "dr": (("doctor",),), "mr": (("mister",),), "mrs": (("missus",),),
    "prof": (("professor",),), "vs": (("versus",),), "st": (("saint",), ("street",)),
    "mg": (("milligrams",), ("milligram",)),
    "kg": (("kilograms",), ("kilogram",)),
    "km": (("kilometers",), ("kilometres",), ("kilometer",)),
    "cm": (("centimeters",), ("centimetres",)),
    "mm": (("millimeters",), ("millimetres",)),
    "ml": (("milliliters",), ("millilitres",)),
    "lb": (("pounds",), ("pound",)), "lbs": (("pounds",),),
    "oz": (("ounces",), ("ounce",)), "ft": (("feet",), ("foot",)),
    "hr": (("hours",), ("hour",)), "sec": (("seconds",), ("second",)),
    # Honorifics and street suffixes. The synthetic corpus writes these in
    # abbreviated form and expects the full word spoken, so every one of them
    # needs a reading here or a correct rendering scores as wrong the moment
    # the recogniser writes the abbreviation back.
    "ms": (("miss",), ("mizz",)), "sgt": (("sergeant",),),
    "capt": (("captain",),), "rev": (("reverend",),),
    "gov": (("governor",),), "sen": (("senator",),),
    "lt": (("lieutenant",),), "col": (("colonel",),),
    "gen": (("general",),), "jr": (("junior",),), "sr": (("senior",),),
    "ave": (("avenue",),), "blvd": (("boulevard",),), "rd": (("road",),),
    "ln": (("lane",),), "ct": (("court",),), "pl": (("place",),),
    "pkwy": (("parkway",),), "sq": (("square",),), "hwy": (("highway",),),
    "apt": (("apartment",),), "ste": (("suite",),),
    "approx": (("approximately",),), "dept": (("department",),),
}


def _roman_to_int(tok: str) -> int | None:
    """Length>=2 only, so the pronoun 'I' is never mistaken for a numeral."""
    if not ROMAN_RE.match(tok):
        return None
    total, prev = 0, 0
    for ch in reversed(tok):
        val = ROMAN[ch]
        total = total - val if val < prev else total + val
        prev = max(prev, val)
    return total


def _expand_currency(text: str) -> str:
    """$10.99 -> '10 dollars 99 cents'; $25 -> '25 dollars'.

    Done before tokenizing so the symbol-side and word-side land on the same
    tokens: the spoken reference already says "ten dollars and ninety nine
    cents".
    """
    for sym, (major, minor) in CURRENCY_SYMBOLS.items():
        def repl(m: re.Match, major=major, minor=minor) -> str:
            whole, frac = m.group(1).replace(",", ""), m.group(2)
            if frac and int(frac.ljust(2, "0")) != 0:
                return f" {whole} {major} {frac.ljust(2, '0')} {minor} "
            return f" {whole} {major} "
        text = re.sub(re.escape(sym) + r"\s*(\d[\d,]*)(?:\.(\d{1,2}))?", repl, text)
    return text


def _tokenize(text: str, expand_currency: bool = True) -> list[str]:
    """Split into words, digit groups, and the symbols an ASR writes back.

    A decimal point inside a number (3.14) survives as part of the token; a
    sentence-final period is dropped; every other symbol in SYMBOL_READINGS is
    emitted as its own token so canonical_variants can offer its readings.

    `expand_currency=False` is used for the WRITTEN side during target
    extraction: expanding "$10.99" into "10 dollars 99 cents" performs exactly
    the verbalization under test, which would erase the target and make every
    Currency item look like it had nothing to check.
    """
    if expand_currency:
        text = _expand_currency(text)
    text = text.lower()
    # A CTC recogniser writes "per cent" as two words; the reference has one.
    # Nothing is lost by joining them: no English sentence needs them apart.
    text = re.sub(r"\bper[ -]cent\b", "percent", text)
    # "u.s.c." -> "usc", matching the spoken "U S C" that
    # _collapse_letters folds the same way. Without this the periods
    # become their own tokens and break the letter run, so a correctly
    # spoken "42 U.S.C. 1983" could not match its own reference. Only
    # runs of SINGLE letters are folded, so "john.doe@example.com" is
    # untouched.
    text = re.sub(r"\b((?:[a-z]\.){2,})",
                  lambda m: m.group(1).replace(".", ""), text)
    text = re.sub(r"(?<=\d)[,](?=\d)", "", text)            # 1,234 -> 1234
    text = re.sub(r"\.(?=\s|$)", " ", text)                 # sentence period
    text = re.sub(r"(?<=\d)\.(?=\d)", "\x00", text)         # protect decimals
    text = re.sub(rf"([{SYMBOL_CHARS}])", r" \1 ", text)    # isolate symbols
    text = text.replace("\x00", ".")
    text = re.sub(r"[^a-z0-9.@/#+=%&*°_:§♯♭\-\s]+", " ", text)
    toks: list[str] = []
    for t in text.split():
        if not t or t in STOPWORDS:
            continue
        toks.extend(_split_alnum(t))
    return toks


_ORDINAL_DIGIT = re.compile(r"^\d+(st|nd|rd|th)$")
_MIXED = re.compile(r"^(?=[a-z0-9]*[a-z])(?=[a-z0-9]*\d)[a-z0-9]+$")


def _split_alnum(tok: str) -> list[str]:
    """'h2o' -> ['h','2','o'], so it can match a spoken 'H two O'.

    Ordinal digit forms ('20th') are left intact -- they are handled as a unit.
    """
    if _ORDINAL_DIGIT.match(tok) or not _MIXED.match(tok):
        return [tok]
    out: list[str] = []
    for part in re.findall(r"\d+|[a-z]+", tok):
        # A SHORT letter run inside a mixed token is spelled aloud: the "oh"
        # of C2H5OH is "O H", the "hgbh" of a VIN is "H G B H". A longer run
        # is a word that happens to carry digits ("covid19"), and spelling
        # that out would be wrong, so the split stops at four characters.
        if part.isalpha() and len(part) <= 4:
            out.extend(part)
        else:
            out.append(part)
    return out


def _parse_number_run(words: list[str]) -> list[tuple[int, bool]]:
    """Arithmetic reading of a run of number words, as (value, is_ordinal) pairs.

    ["one","two","three"]              -> [(1,F),(2,F),(3,F)]
    ["one","hundred","twenty","three"] -> [(123,F)]
    ["twenty","twenty","three"]        -> [(20,F),(23,F)]
    ["two","thousand","twenty","four"] -> [(2024,F)]
    ["twentieth"]                      -> [(20,T)]
    ["third","nineteen","ninety","five"] -> [(3,T),(19,F),(95,F)]

    Ordinality is per-number, not per-run: "July third nineteen ninety five"
    is one contiguous run in which only the first number is ordinal.
    """
    out: list[tuple[int, bool]] = []
    current = 0        # group being built
    total = 0          # accumulated across thousand+ scales
    started = False
    is_ordinal = False        # ordinality of the CURRENT group only
    any_ordinal = False

    # `has_value` distinguishes "this group is empty" from "this group's value
    # is genuinely zero". Without it, "one zero zero" loses a zero, because a
    # current of 0 looks identical to a group that has not started.
    has_value = False

    def flush() -> None:
        nonlocal current, total, started, is_ordinal, has_value
        if started:
            out.append((total + current, is_ordinal))
        current, total, started, is_ordinal, has_value = 0, 0, False, False, False

    def add_unit(val: int) -> None:
        """A unit continues the group only where English allows it."""
        nonlocal current, started, has_value
        if not started or not has_value:
            current, started, has_value = val, True, True
        elif current >= 100 and current % 10 == 0:
            current += val                      # 'one hundred (twenty) three'
        elif 20 <= current < 100 and current % 10 == 0:
            current += val                      # 'twenty three'
        else:
            flush()
            current, started, has_value = val, True, True   # 'one two three'

    def add_tens(val: int) -> None:
        nonlocal current, started, has_value
        if not started or not has_value:
            current, started, has_value = val, True, True
        elif current >= 100 and current % 100 == 0:
            current += val                      # 'one hundred twenty'
        else:
            flush()
            current, started, has_value = val, True, True    # 'twenty twenty'

    for w in words:
        if w in ORDINAL_UNITS:
            add_unit(ORDINAL_UNITS[w]); is_ordinal = True; any_ordinal = True
        elif w in ORDINAL_TENS:
            add_tens(ORDINAL_TENS[w]); is_ordinal = True; any_ordinal = True
        elif w in ORDINAL_SCALES:
            current = (current or 1) * ORDINAL_SCALES[w]
            started = True; has_value = True; is_ordinal = True; any_ordinal = True
        elif w in SCALES:
            scale = SCALES[w]
            if scale == 100:
                current = (current or 1) * 100
                has_value = True
            else:
                total += (current or 1) * scale
                current = 0
                has_value = False       # 'two thousand' + a fresh group after it
            started = True
        elif w in TENS:
            add_tens(TENS[w])
        elif w in UNITS:
            add_unit(UNITS[w])
    flush()
    return out


def _segmentations(nums: list[tuple[int, bool]]) -> set[tuple[str, ...]]:
    """Every way to read a number sequence as separate or run-together values.

    [1,2,3] yields ("1","2","3"), ("12","3"), ("1","23") and ("123") -- so
    "one two three" matches a hypothesis of either "123" or "1 2 3", which is
    the multiple-valid-verbalizations problem. A segment longer than one may
    not contain an ordinal ("third nineteen" is never "319").
    """
    n = len(nums)
    if n == 0:
        return set()
    if n > 8:  # combinatorial guard; fall back to the two extremes
        sep = tuple(f"{v}#o" if o else str(v) for v, o in nums)
        out = {sep}
        if not any(o for _, o in nums):
            out.add(("".join(str(v) for v, _ in nums),))
        return out

    out: set[tuple[str, ...]] = set()

    def walk(i: int, acc: tuple[str, ...]) -> None:
        if i == n:
            out.add(acc)
            return
        for j in range(i + 1, n + 1):
            seg = nums[i:j]
            if len(seg) == 1:
                v, o = seg[0]
                walk(j, acc + (f"{v}#o" if o else str(v),))
            elif not any(o for _, o in seg):
                walk(j, acc + ("".join(str(v) for v, _ in seg),))

    walk(0, ())
    return out


def _run_variants(words: list[str]) -> set[tuple[str, ...]]:
    """Acceptable canonical forms for one numeric run."""
    if "point" in words:
        idx = words.index("point")
        left = _parse_number_run(words[:idx])
        right = _parse_number_run(words[idx + 1:])
        whole = "".join(str(v) for v, _ in left) or "0"
        frac = "".join(str(v) for v, _ in right)
        return {(f"{whole}.{frac}",)}

    nums = _parse_number_run(words)
    if not nums:
        return {tuple(words)}
    out = _segmentations(nums)
    # "one third" / "two fifths": a cardinal followed by a trailing ordinal is
    # also a fraction reading, so it can match a written "1/3".
    if len(nums) >= 2 and nums[-1][1] and not any(o for _, o in nums[:-1]):
        head = tuple(str(v) for v, _ in nums[:-1])
        out.add(head + (f"{nums[-1][0]}f",))
    return out


def _digit_token(tok: str) -> str | None:
    """Normalize a digit-side token: '20th' -> '20#o', '2023' -> '2023'."""
    m = re.fullmatch(r"(\d+)(st|nd|rd|th)", tok)
    if m:
        return f"{int(m.group(1))}#o"
    if re.fullmatch(r"\d+\.\d+", tok):
        return tok
    if re.fullmatch(r"\d+", tok):
        return str(int(tok))
    return None


def _collapse_letters(toks: list[str]) -> list[str]:
    """'u s a' -> 'usa' so spelled-out initialisms match written ones."""
    out: list[str] = []
    run: list[str] = []
    for t in toks:
        if len(t) == 1 and t.isalpha():
            run.append(t)
        else:
            if len(run) >= 2:
                out.append("".join(run))
            elif run:
                out.append(run[0])
            run = []
            out.append(t)
    if len(run) >= 2:
        out.append("".join(run))
    elif run:
        out.append(run[0])
    return out


def canonical_variants(text: str, merge_digits: bool = False,
                       expand_currency: bool = True) -> set[tuple[str, ...]]:
    """All acceptable canonical token sequences for `text`.

    `merge_digits` collapses adjacent digit tokens, so "978 3 16 148410 0" and a
    digit-by-digit reading agree. Turn it on ONLY for identifier categories
    (ISBN, phone, serial, version). Leaving it off elsewhere is deliberate:
    with it on, "two zero two three" would match "twenty twenty three", and
    reading a year digit-by-digit is exactly the failure this benchmark exists
    to catch.
    """
    toks = _tokenize(text, expand_currency=expand_currency)
    segments: list[list[tuple[str, ...]]] = []
    buf: list[str] = []
    run: list[str] = []

    def close_run() -> None:
        nonlocal run
        if run:
            segments.append(sorted(_run_variants(run)))
            run = []

    def close_buf() -> None:
        nonlocal buf
        if buf:
            segments.append([tuple(_collapse_letters(buf))])
            buf = []

    i = 0
    while i < len(toks):
        tok = toks[i]

        # "3/4" -> also readable as the fraction "three quarters"
        if (i + 2 < len(toks) and toks[i + 1] == "/"
                and re.fullmatch(r"\d+", tok) and re.fullmatch(r"\d+", toks[i + 2])):
            close_run(); close_buf()
            a, b = int(tok), int(toks[i + 2])
            segments.append([(str(a), f"{b}f"), (str(a), "slash", str(b))])
            i += 3
            continue

        if tok in NUMBER_WORDS:
            close_buf()
            run.append(tok)
            i += 1
            continue

        close_run()

        if tok in FRACTION_WORDS:
            close_buf()
            segments.append([(f"{FRACTION_WORDS[tok]}f",)])
            i += 1
            continue

        # "40.7128° N" is read "forty point seven one two eight degrees north".
        # Scoped to a compass letter directly after a degree sign, because the
        # bare letters n/s/e/w must keep their ordinary behaviour: putting them
        # in ABBREVIATIONS would break the letter run in "U S A", which is a
        # far more common pattern than a coordinate.
        if (tok == "°" and i + 1 < len(toks)
                and toks[i + 1] in COMPASS_LETTERS):
            close_buf()
            segments.append([(r,) for r in SYMBOL_READINGS[tok]])
            segments.append([(toks[i + 1],), (COMPASS_LETTERS[toks[i + 1]],)])
            i += 2
            continue

        if tok in SYMBOL_READINGS:
            close_buf()
            readings = SYMBOL_READINGS[tok]
            if merge_digits and tok in IDENTIFIER_SYMBOL_READINGS:
                readings = IDENTIFIER_SYMBOL_READINGS[tok]
            segments.append([(r,) if r else () for r in readings])
            i += 1
            continue

        d = _digit_token(tok)
        if d is not None:
            close_buf()
            segments.append([(d,)])
            i += 1
            continue

        if tok in ABBREVIATIONS:
            close_buf()
            segments.append([(tok,), *ABBREVIATIONS[tok]])
            i += 1
            continue

        rn = _roman_to_int(tok)
        if rn is not None:
            close_buf()
            segments.append([(str(rn),), (tok,)])
            i += 1
            continue

        buf.append(tok)
        i += 1
    close_run()
    close_buf()

    if not segments:
        return {()}

    combos = itertools.islice(itertools.product(*segments), MAX_VARIANTS)
    out = {tuple(itertools.chain.from_iterable(c)) for c in combos}
    if merge_digits:
        out |= {_merge_digit_runs(v) for v in out}
    return out


def _merge_digit_runs(variant: tuple[str, ...]) -> tuple[str, ...]:
    out: list[str] = []
    run: list[str] = []
    for tok in variant:
        if tok.isdigit():
            run.append(tok)
        else:
            if run:
                out.append("".join(run)); run = []
            out.append(tok)
    if run:
        out.append("".join(run))
    return tuple(out)


def _wer(ref: Iterable[str], hyp: Iterable[str]) -> float:
    r, h = list(ref), list(hyp)
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rt in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, ht in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rt != ht))
        prev = cur
    return prev[len(h)] / len(r)


# Categories where audible digit grouping carries no meaning, so a
# digit-by-digit reading is a correct rendering rather than an error.
IDENTIFIER_CATEGORIES = frozenset({
    "ISBN", "Phone Number", "License Plate or Serial Numbers",
    "Version Numbers", "Vehicle or Product Code", "Geographic Coordinates",
})


def merge_digits_for(category: str | None) -> bool:
    return category in IDENTIFIER_CATEGORIES if category else False


def best_wer(hypothesis: str, reference: str, category: str | None = None) -> float:
    """Lowest WER over every (reference variant, hypothesis variant) pair."""
    md = merge_digits_for(category)
    refs = canonical_variants(reference, md)
    hyps = canonical_variants(hypothesis, md)
    return min(_wer(r, h) for r in refs for h in hyps)


def matches(hypothesis: str, reference: str, category: str | None = None) -> bool:
    """True when some canonical reading of each side agrees exactly.

    Whole-sentence match. Strict, and sensitive to ASR noise anywhere in the
    carrier -- wav2vec2 writing "DEAD LINE" for "deadline" fails this even
    though the verbalization under test was perfect. Prefer target_match for
    scoring; keep this as the strict secondary number.
    """
    md = merge_digits_for(category)
    return not canonical_variants(reference, md).isdisjoint(
        canonical_variants(hypothesis, md))


# Bound the variant cross-product. The cap is applied AFTER sorting, so it is
# not a random sample -- it deterministically keeps the same subset every run.
# It was 48, which silently broke every hyphenated identifier: an ISBN with
# four hyphens produces ~150 readings and the one that matches sorted last.
# The work per pair is a Counter subtraction, so 256 costs nothing measurable
# on a 4,000-record score and removes the truncation as a source of false
# negatives.
_PAIR_CAP = 256
_TARGET_CAP = 64


def _surface_tokens(text: str) -> list[str]:
    """Whitespace tokens with edge punctuation stripped, nothing normalized.

    Internal structure is preserved on purpose: "10:30", "$10.99" and
    "05/20/2023" each stay a SINGLE token, so a diff against the spoken form
    aligns one written token against the several words that verbalize it.
    """
    out = []
    for tok in text.lower().split():
        tok = tok.strip(".,;:!?\"'()[]{}")
        if tok:
            out.append(tok)
    return out


def target_text(written: str, spoken: str) -> str:
    """The words the spoken form uses in place of the non-standard written token.

    Found by SURFACE diff, not canonical diff. Canonicalization deliberately
    equates "10:30" with "ten thirty", so a canonical diff reports no target
    for over half the corpus -- it erases the very transformation under test.
    Surface diff locates the span; canonicalization then scores it.

        "Let's meet at 10:30 tomorrow."  ->  "ten thirty"
        "The price is $10.99."           ->  "ten dollars and ninety nine cents"
        "The deadline is 2024-12-31."    ->  "December thirty first twenty twenty four"
    """
    import difflib
    w, s = _surface_tokens(written), _surface_tokens(spoken)
    sm = difflib.SequenceMatcher(a=w, b=s, autojunk=False)
    parts: list[str] = []
    for tag, _i1, _i2, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "insert"):
            parts.extend(s[j1:j2])
    return " ".join(parts)


def target_tokens(written: str, spoken: str,
                  category: str | None = None) -> list["Counter[str]"]:
    """Canonical readings of the target span, smallest first."""
    from collections import Counter
    tgt = target_text(written, spoken)
    if not tgt:
        return []
    md = merge_digits_for(category)
    out = [Counter(v) for v in sorted(canonical_variants(tgt, md))[:_TARGET_CAP]]
    out.sort(key=lambda c: sum(c.values()))
    return out


def has_target(written: str, spoken: str, category: str | None = None) -> bool:
    """False when the two forms canonicalize to the same token multiset.

    "They married on 3rd July 1995." vs "...on July third nineteen ninety five."
    differ only in word ORDER, so after canonicalization there is no token the
    TTS had to add. Such an item cannot distinguish a correct rendering from a
    wrong one, so it is excluded and counted rather than guessed at -- scanning
    for some larger target that happens to fit is how a scorer starts inventing
    results.
    """
    targets = target_tokens(written, spoken, category)
    return bool(targets) and bool(targets[0])


def _singular(tok: str) -> str:
    """Fold a trailing plural s, for comparison only.

    "THE WEIGHT LIMIT IS FIFTY POUND" is a recogniser dropping an s, not a
    system saying the wrong number, and the benchmark measures the number.
    NUMBER_WORDS are exempt: "hundred" and "hundreds" carry different values
    and must never be folded together.
    """
    if tok in NUMBER_WORDS or len(tok) <= 3:
        return tok
    if tok.endswith("ss") or not tok.endswith("s"):
        return tok
    return tok[:-1]


def _fold(counter: "Counter[str]") -> "Counter[str]":
    from collections import Counter
    return Counter(_singular(t) for t in counter.elements())


def target_match(hypothesis: str, written: str, spoken: str,
                 category: str | None = None) -> bool:
    """Did the hypothesis contain the required verbalization?

    Scores the non-standard token only, so ASR errors elsewhere in the carrier
    ("DEAD LINE" for "deadline") cannot mark a correct rendering wrong.
    Order-free within the target, because "July third" and "third July" are
    both correct readings of "3rd July".

    EVERY reading of the target span counts, not just the shortest. The
    shortest reading of "four fifty seven" is the single number 457, which
    a recognizer writing "4:57" never produces -- preferring it marked 28
    of 40 correctly spoken times as wrong. A longer reading is a STRICTER
    requirement (more tokens the hypothesis must contain), so admitting
    all of them can only turn a false miss into a hit, never the reverse.
    Call has_target() first; this returns False for a no-target item.
    """
    from collections import Counter
    candidates = target_tokens(written, spoken, category)
    if not candidates or not candidates[0]:
        return False
    md = merge_digits_for(category)
    hyps = [Counter(h) for h in sorted(canonical_variants(hypothesis, md))[:_PAIR_CAP]]
    if any(not (t - h) for t in candidates for h in hyps):
        return True
    # Retry with plurals folded on BOTH sides. Done as a fallback rather than
    # up front so an exact match is never reinterpreted, and so the looser
    # comparison can only ever turn a miss into a hit, never the reverse.
    folded = [_fold(h) for h in hyps]
    return any(not (_fold(t) - h) for t in candidates for h in folded)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    pairs = [
        ("The event is on May 20th, 2023.", "The event is on May twentieth twenty twenty three."),
        ("The price is $10.99.", "The price is ten dollars and ninety nine cents."),
        ("The address is 123 Main St, USA.", "The address is one two three Main St, U S A."),
    ]
    for hyp, ref in pairs:
        print(f"{matches(hyp, ref)!s:>5}  wer={best_wer(hyp, ref):.2f}  {hyp}")
