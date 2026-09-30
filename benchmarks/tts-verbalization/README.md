# TTS verbalization benchmark

**Do open text-to-speech models say dates, prices and phone numbers correctly?**

Seven open models, 1,080 generated sentences, 27 categories, two independent
speech recognizers.

<!-- verify:headline -->
| model | correct | 95% CI | compute/clip |
|---|---|---|---|
| kokoro-82m | 0.61 | 0.58-0.64 | 0.31s |
| orpheus-3b | 0.54 | 0.51-0.57 | 5.50s |
| qwen3-tts | 0.49 | 0.46-0.52 | 4.99s |
| chatterbox-turbo | 0.49 | 0.46-0.52 | 1.06s |
| chatterbox | 0.44 | 0.41-0.47 | 3.48s |
| chatterbox-multilingual | 0.25 | 0.23-0.28 | 2.40s |
| parler-tts | 0.03 | 0.02-0.04 | 3.06s |
<!-- /verify:headline -->

Scored with OpenAI whisper-1. The smallest model is the most accurate and uses
roughly 18x less compute than the runner-up. No model can say a hashtag: 0 correct
out of 279. The best date score is 0.23.

**Read the full write-up:**
[dograh.com/hub/experiments/tts-verbalization-benchmark](https://dograh.com/hub/experiments/tts-verbalization-benchmark)

**Reproduce it.** Open your AI agent (Claude Code or Cursor) and paste this:

> Reproduce Dograh's TTS verbalization experiment: run `git clone --depth 1 https://github.com/RuchiketJadhav/experiments.git && cd experiments/benchmarks/tts-verbalization`, then follow AGENTS.md. Start with the free verification, and ask me before spending anything.

Tier 1 is free, needs no API keys, and takes about a minute. It recomputes every
number on this page from `results/records.jsonl.gz` and fails if any of them has
drifted. See [METHODOLOGY.md](METHODOLOGY.md) for how the experiment was run.

---

The rest of this page is the same write-up that appears on dograh.com, kept here
so the repo stands on its own.

---

## The question

Does a text-to-speech model **say** the right words?

Not how natural it sounds. Not audio quality. If the text says `05/20/2023`, does
the audio say "May twentieth, twenty twenty-three", or does it say "five, twenty,
twenty twenty-three"?

For a voice agent confirming a booking, quoting a price or reading back a phone
number, that single behaviour decides whether the call works. Easy sentences have
been a solved baseline for decades, so every sentence in this corpus is built
around something that breaks.

## Headline findings

**The smallest model is the most accurate.** kokoro-82m is 82 million parameters
and scores 0.61. orpheus-3b is 3 billion and scores 0.54. kokoro also uses
roughly 18x less compute per clip.

**No model can say a hashtag.** Hashtag or Mention scores 0 correct out of 279
clips under whisper-1 and 0 out of 278 under nova-3. Not "scored badly" — zero,
under both recognizers.

**Dates are broken everywhere.** The best score any model reached is 0.23, by
chatterbox. Three out of four dates come out wrong.

**Two recognizers rank the models identically and still disagree on 11% of
individual clips.** The ranking is trustworthy. Any single per-clip verdict is not.

## Results

Match rate is the fraction of clips where the required verbalization was present.
whisper-1 is the primary scorer; nova-3 is a cross-check, never a scorer (see
[Why nova-3 is not a scorer](#why-nova-3-is-not-a-scorer)).

<!-- verify:results -->
| model | whisper-1 | 95% CI | nova-3 |
|---|---|---|---|
| kokoro-82m | 0.61 | 0.58-0.64 | 0.62 |
| orpheus-3b | 0.54 | 0.51-0.57 | 0.48 |
| qwen3-tts | 0.49 | 0.46-0.52 | 0.45 |
| chatterbox-turbo | 0.49 | 0.46-0.52 | 0.45 |
| chatterbox | 0.44 | 0.41-0.47 | 0.39 |
| chatterbox-multilingual | 0.25 | 0.23-0.28 | 0.24 |
| parler-tts | 0.03 | 0.02-0.04 | 0.03 |
<!-- /verify:results -->

`qwen3-tts` and `chatterbox-turbo` are a statistical tie: 0.4929 against 0.4926,
with intervals that overlap almost completely. Do not read an ordering between
them.

`chatterbox-multilingual` is second to last. `parler-tts` is last.

### Compute per clip

Median billed prediction time on Replicate, measured on this corpus. This came
from a separate 10-clip timing run per model (`results/latency-manifest.json`),
not from the scored run, so treat it as an order-of-magnitude figure rather than
a precise one. `verify.py` does not check these numbers.

| model | billed compute | wall clock |
|---|---|---|
| kokoro-82m | 0.31s | 0.93s |
| chatterbox-turbo | 1.06s | 1.47s |
| chatterbox-multilingual | 2.40s | 3.10s |
| parler-tts | 3.06s | 3.40s |
| chatterbox | 3.48s | 3.98s |
| qwen3-tts | 4.99s | 5.38s |
| orpheus-3b | 5.50s | 8.13s |

kokoro-82m against orpheus-3b is roughly 18x on billed compute, at n=10 per model.

## Best score per category

The best any of the seven models reached, under whisper-1.

<!-- verify:categories -->
| category | best | model |
|---|---|---|
| Hashtag or Mention | 0.00 | chatterbox-multilingual |
| Date | 0.23 | chatterbox |
| Version Numbers | 0.30 | kokoro-82m |
| Fractions | 0.40 | qwen3-tts |
| Biological Classification | 0.46 | orpheus-3b |
| URL or Email | 0.50 | chatterbox-turbo |
| ISBN | 0.53 | orpheus-3b |
| Address | 0.55 | kokoro-82m |
| Phone Number | 0.60 | chatterbox |
| Chemical Formula | 0.62 | chatterbox |
| Geographic Coordinates | 0.67 | qwen3-tts |
| Mathematical Expression | 0.70 | qwen3-tts |
| Unit | 0.72 | qwen3-tts |
| Time | 0.72 | kokoro-82m |
| Legal Reference | 0.78 | kokoro-82m |
| License Plate or Serial Numbers | 0.78 | chatterbox |
| Cardinal | 0.82 | orpheus-3b |
| Abbreviation | 0.85 | kokoro-82m |
| Roman Numeral | 0.88 | kokoro-82m |
| Decimal | 0.90 | kokoro-82m |
| Sports score | 0.90 | qwen3-tts |
| Vehicle or Product Code | 0.95 | kokoro-82m |
| Currency | 0.97 | kokoro-82m |
| Ordinal | 0.97 | orpheus-3b |
| Initialism or Acronym | 1.00 | kokoro-82m |
| Musical Notation | 1.00 | kokoro-82m |
| Stock Ticker | 1.00 | kokoro-82m |
<!-- /verify:categories -->

Full detail, per recognizer and per model with confidence intervals, is in
`results/category-results.csv`.

**The spread inside one model is wider than the spread between models.** kokoro
scores 1.00 on Musical Notation and 0.00 on Hashtag in the same run. A 27-category
average is a headline, not a decision. If your product reads order numbers, look at
License Plate. If it books appointments, look at Date.

## What the hashtag zero actually is

The zero is real, and it is not a word-splitting failure.

The models handle the splitting reasonably. kokoro's rendering of
`Follow #GreenBreak for updates during the conference.` was transcribed as
"Follow GreenBreak for updates during the conference", and
`The trending topic this week is #WorldScience again.` came back as
"the trending topic this week is world science again" — split correctly.

What no model ever does is **say the symbol**. Across every whisper-1 transcript of
a `#` item, the word "hashtag" appears zero times. Across every transcript of an
`@` item, "underscore" appears zero times.

The reference reading in this corpus includes the symbol: `#TravelTips` is
"hashtag travel tips", `@alice_dev` is "at alice underscore dev". So an item fails
even when the words either side are perfect.

That is a judgement call in the corpus, and worth stating plainly: plenty of people
read a hashtag aloud without saying "hashtag". If your product does the same, treat
this category as measuring a convention you may not share, and read the other 26.

## What parler-tts at 0.03 actually is

parler-tts is not a model with clear speech and a missing normalization layer. Its
speech is substantially broken.

Its mean word error rate under whisper-1 is **0.36**, against **0.07** for
kokoro-82m. It garbles whole sentences, not just the hard span:

| written | whisper-1 heard |
|---|---|
| Her birthday falls on 2009-02-26 this year. | Herbal tea flowers enswore so, very gee. Thok! |
| They signed the agreement on 01/26/1952 in Boston. | They signed the Agreement on Fairly Fragile Figures in Boston. |
| The museum reopened on 4th June 2014 after the renovation. | The museum reopened on Payback June. |

Read its 0.03 as a floor that shows the benchmark discriminates, not as a claim
about any specific missing component. We did not inspect its frontend and make no
claim about one.

## Why nova-3 is not a scorer

A single recognizer deciding what a model "said" is a weak foundation, and there is
published work showing TTS rankings can flip across recognizer families
(arXiv:2607.08256). So every clip went through two.

Deepgram nova-3 is a cross-check only. It failed our validation gate as a scorer
because its language model expands abbreviations — it turned "mph" into "miles per
hour", which repairs a real verbalization failure into a pass. `numerals=false`
stops digit formatting but not lexical expansion. The reasoning is in the comment
in `configs/replicate_open10.yaml`.

The two recognizers rank all seven models **identically**. They disagree on **832 of
7,497** clips scored by both, which is 11.1%. Of those, whisper-1 passes where
nova-3 fails on 525, and the reverse on 307.

Both facts matter and they point different ways. The ranking is solid: two
independent graders, complete agreement. A single clip's verdict is not: roughly one
in nine is contested, and only listening settles those. Every contested clip is in
`results/recognizer-disagreements.csv` with what each recognizer heard.

## The corpus

<!-- verify:facts -->
- `unique_records` = `15120`
- `records_scored` = `15051`
- `records_asr_error` = `47`
- `records_screened_transcript` = `22`
- `items` = `1080`
- `categories` = `27`
- `models` = `7`
- `common_clips` = `7497`
- `disagreements` = `832`
- `hashtag_scored_whisper` = `279`
- `hashtag_correct_whisper` = `0`
<!-- /verify:facts -->

1,080 sentences, 27 categories, 40 each, **generated rather than written**. Each
category has a generator in `gen_corpus.py` that produces the written form and its
correct spoken form from the same underlying value, so the reference reading is
correct by construction. No model guessed at an answer and no human adjudicated a
judgement call.

Every candidate had to pass four checks before entering the corpus:

- the written and spoken forms genuinely differ, so the item can tell right from wrong
- the correct reading scores as correct
- a deliberately corrupted reading scores as **wrong**, so the item detects an error
- the spoken form cannot be long enough to trip the audio screening rules

The corpus regenerates from a seed and a spec file:

```bash
python gen_corpus.py --locale en-US        # deterministic; same seed, same corpus
```

`data/spec/en.yaml` holds the carrier sentences and per-category counts. That file
is the language seam: a second language is a new spec plus a new generator table.

**A note on the corpus checksum.** `dataset_sha256` in `results/manifests/` is
`11b71377...`, the hash of the corpus file as it existed on the machine that ran
the study: generated on Windows, so CRLF line endings. The file shipped here is
the same 1,080 items with LF endings and hashes `c34716bb...`. The content is
byte-identical once line endings are normalised; we checked item by item.

`gen_corpus.py` now pins its output to LF on every platform. It previously used
Python's default newline translation, which meant the same seed produced different
bytes on Windows and Linux. A corpus generator whose output depends on the
operating system is not reproducible, so that is fixed rather than documented.

### Of the 15,120 records

47 are `asr_error`, all HTTP 429 rate limits from the recognizer, and 22 are
`screened_transcript` — 13 empty transcripts and 9 flagged as recognizer
hallucinations, where the transcript ran more than twice the reference length.
Screened clips are reported as their own outcome and excluded from the score rather
than folded in as bad results. A broken pipeline should not look like a mediocre
voice.

## Limitations

- **This is not a word error rate.** Every sentence is chosen to be hard. Read a
  score as "how often does this model get the difficult part right", never as
  general accuracy.
- **These are models as packaged on Replicate**, not bare checkpoints. Wrapper
  configuration measurably changes results. Read each row as "this model, as
  shipped", with the revision pinned in `results/manifests/`.
- **English only.**
- **The hashtag convention is ours.** See above.
- **Compute figures come from 10 clips per model**, not the scored run.
- **69 of 15,120 records (0.5%) never produced a score** and are excluded, not
  guessed at.

## Reproduce

Three tiers. `AGENTS.md` is the runbook; it always starts with Tier 1 and always
stops for your approval before spending anything.

### Tier 1 — Verify (free, no keys, about a minute)

```bash
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python verify.py
```

Recomputes every summary from `results/records.jsonl.gz` with this study's own
`score.py`, compares them exactly to the published summary files, and checks every
number printed on this page against the recomputed value. Prints a pass/fail table
and exits non-zero on any mismatch.

This is the check that matters. It proves the numbers above were computed from the
data in this repo, not typed in by hand.

### Tier 2 — Smoke (paid, needs API keys)

About 20 items sampled across all categories, through the real models and both
recognizers. Reports actual spend and time, then projects the full run from them.

### Tier 3 — Full (paid)

7,560 syntheses and 15,120 transcriptions. Roughly 24 hours serially, or 4-6 hours
sharded by model with `deploy/shard_by_model.sh`. The machine barely matters: the
time is spent waiting on APIs, not computing. See `deploy/README.md`.

A fresh Tier 3 run will **not** match these numbers to the decimal. Hosted models
change behind the same version tag and sampling adds noise. See `AGENTS.md` for the
size of difference to expect.

## Files

```
README.md                 this page
AGENTS.md                 the runbook an agent follows
verify.py                 Tier 1
gen_corpus.py             generates the corpus from data/spec/en.yaml
canonicalize.py           the metric: target extraction and reading equivalence
run.py                    the driver: resumable, content-hash audio cache
score.py                  rates, bootstrap CIs, exclusion accounting
disagree.py               cross-recognizer disagreement
audit.py                  blind listening audit (needs audio, not published)
data/spec/en.yaml         corpus spec: carriers and counts
data/synthetic/en-US.jsonl the 1,080 items
configs/                  model and recognizer configs
deploy/                   AWS runbook and sharding
results/                  the published run
tests/                    the test suite
```

## Licence

Code BSD-2-Clause (`../../LICENSE`). Data and results CC BY 4.0
(`../../LICENSE-DATA`). The corpus is generated by `gen_corpus.py` from a seed and
a spec file, so no third-party rights attach to any row.

`data/loaders/polynorm.py` and `configs/replicate_open10.yaml` reference Apple's
PolyNorm-Bench, which is CC BY-NC-ND. That code downloads the dataset at run time.
No PolyNorm row, transcript or result appears anywhere in this repository.
