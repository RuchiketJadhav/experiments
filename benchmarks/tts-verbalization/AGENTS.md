# Runbook: TTS verbalization study

You are an agent reproducing this study for someone who has just cloned the repo
and knows nothing about it. Work through this file in order.

**Two rules that override anything else here.**

1. **Never ask the user to paste an API key into the chat.** Tell them to copy
   `.env.example` to `.env` and fill it in themselves. Check only that the
   variables are set, never their values, and never print a value.
2. **Never spend their money without asking.** Tier 1 is free and you run it
   unprompted. Tiers 2 and 3 cost real money: explain what it will cost and how
   you arrived at that estimate, then STOP and wait for an explicit yes.

---

## Step 1: environment

Detect the operating system and the Python version. Python 3.10 or newer is
required. No ffmpeg is needed: `soundfile` reads wav and mp3 through libsndfile.

```bash
python --version                     # or python3 --version
python -m venv .venv
. .venv/bin/activate                 # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

On Linux, `soundfile` needs libsndfile. If the import fails:
`sudo apt-get install -y libsndfile1`.

Report the OS, the Python version and whether the install succeeded.

## Step 2: tests

```bash
bash run_tests.sh
```

Use the script, not a bare loop over `tests/*.py`. A loop prints a failing suite
and then exits 0, which reports a broken checkout as healthy.

All suites should pass with no network and no API key. If any fail, stop and show
the output. Do not continue to Tier 1 with a failing suite: `verify.py` shares the
scoring code the tests cover.

## Step 3: Tier 1 — Verify (free, always run this first)

```bash
python verify.py
```

This recomputes every published summary from `results/records.jsonl.gz` using this
study's own `score.py`, compares them exactly to `results/summary-*.json`, and
checks every number printed in `README.md` against the recomputed value. It prints
a pass/fail table and exits non-zero on any mismatch.

**Show the user the full table.** Then say plainly what it proves: the numbers in
the README were computed from the data in this repo, not typed in by hand. It does
not prove the models behave this way today, only that the published analysis is
honest about the published data.

If anything fails, stop and show which check failed and by how much. A failure
means either the data or the README has been changed.

**For most people this is the whole job.** Do not push them toward a paid tier. Ask
whether they want to go further before describing Tiers 2 or 3 in any detail.

## Step 4: keys, only if they want a paid tier

```bash
cp .env.example .env
```

Tell the user to open `.env` and fill in the three values themselves. Then verify
only presence:

```bash
python -c "
import os
for k in ('REPLICATE_API_TOKEN','OPENAI_API_KEY','DEEPGRAM_API_KEY'):
    print(f'{k}: {\"set\" if os.environ.get(k) else \"NOT SET\"}')
"
```

Load them into the environment before running anything:
`set -a; . ./.env; set +a` (bash) or the PowerShell equivalent.

Never echo a value. If one is missing, say which, and stop.

## Step 5: Tier 2 — Smoke (paid; ask first)

**Before running, tell the user this, and wait for a yes.**

> This synthesises 20 sentences on each of 7 models and transcribes every clip with
> 2 recognizers: 140 syntheses and 280 transcriptions. Based on the full run, which
> cost about US$20 for 7,560 syntheses, 140 syntheses is roughly US$0.40, plus about
> US$0.15 of transcription. Call it under a dollar. It takes 10 to 20 minutes,
> mostly waiting on cold starts. I will report what it actually cost when it
> finishes.

That estimate is derived, not quoted from a price list: US$20 over 7,560 syntheses
is about US$0.0026 each, and the cheap models finish first so a sample skews
cheaper. Say where the number comes from, because a stranger has no reason to
trust it.

```bash
python run.py --config configs/synthetic_en.yaml --out runs/smoke --sample 20
python score.py --run runs/smoke --asr whisper
```

Use `--sample`, never `--limit`. The corpus is ordered by category, so `--limit 20`
gives 20 Date items and tells you nothing; `--sample 20` spreads across all 27.

Afterwards, report the real spend from the provider dashboards and the real elapsed
time, then project the full run from those two measured numbers rather than from
the estimate above.

## Step 6: Tier 3 — Full (paid; ask first)

**Before running, tell the user this, and wait for a yes.**

> The full run is 7,560 syntheses and 15,120 transcriptions. Our run cost about
> US$20 on Replicate plus roughly US$8 of transcription. Serially it takes about 24
> hours. Sharded by model it takes 4 to 6 hours, bounded by the slowest model.

```bash
bash deploy/shard_by_model.sh configs/synthetic_en.yaml runs/full
```

One process per model, merged and scored at the end. `deploy/README.md` has the AWS
path: the run is entirely network-bound, so a small instance is correct and a GPU is
not. Machine choice barely affects the time.

Two things that cost us days and are worth passing on:

- Run it inside `tmux` or `screen`. An SSH disconnect kills the parent, and the
  parent is what merges and scores at the end.
- Keep an eye on provider credit. Running out mid-run marks items as failed rather
  than pausing, and those items then need another pass.

## Step 7: compare honestly

After Tier 2 or Tier 3, compare against `results/summary-whisper-1.json` and say
plainly what matches and what does not.

**A fresh run will not match to the decimal, and that is expected.** Two reasons:

- **Hosted models change behind a fixed version tag.** Every model here is pinned by
  a Replicate version hash in `results/manifests/`, but the served weights and the
  wrapper can change under it.
- **Sampling noise.** At n=1,080 per model the 95% interval spans roughly plus or
  minus 3 points.

So:

- **Within about 3 points** on a model: consistent with this study. Say so.
- **More than 5 points**, or a change in the ORDER of the models: a real difference.
  Report it as a finding, not as an error. Check the version hashes first.
- **Tier 2 will swing much more.** At n=20 per model the interval is roughly plus or
  minus 20 points. Do not read a ranking from a smoke run, and say that clearly.

Never adjust, round or explain away a difference to make it match. The point of the
repo is that someone can disagree with us using our own tools.

## If something breaks

- **Import errors on `soundfile`**: install libsndfile (above).
- **HTTP 402 from Replicate**: out of credit. Top up; finished items are cached and
  skipped, so you only pay for the remainder.
- **HTTP 429**: rate limited. Lower `MAX_PARALLEL`, and raise the retry budget with
  `TTSBENCH_RETRY_ATTEMPTS=10 TTSBENCH_RETRY_BACKOFF=3`.
- **An empty-bodied HTTP 400 from a recognizer**: usually a malformed header, most
  often a key with a stray character. Check the key's length and that it contains no
  whitespace or escape characters.
- **A run stops partway**: it is resumable. Re-run the same command. Audio and
  transcripts are content-addressed and cached; add `--retry-errors` to re-attempt
  items whose recorded outcome was a transient failure.
