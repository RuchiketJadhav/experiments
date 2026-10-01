# Adding an experiment

An experiment is a folder under `benchmarks/` (comparing models) or `tests/` (one thing in depth) that a stranger can reproduce with one
pasted prompt. That is the whole bar. If someone with no context and no access to
your machine cannot get from the prompt to your numbers, it is not ready.

## Start from the template

```bash
cp -r _template benchmarks/<your-experiment>     # comparing models
# or: cp -r _template tests/<your-experiment>   # testing one thing in depth
cd benchmarks/<your-experiment>
```

Fill in `README.md` and `AGENTS.md`. Both skeletons say what each section is for.

## What a experiment must have

**`README.md`** — for humans. The question, the headline findings, results tables,
the method, the limitations, and how to reproduce. Every number in it must be
reproduced by `verify.py`.

**`AGENTS.md`** — for agents. A runbook: detect the environment, install, run the
tests, run Tier 1, and stop for approval before any paid tier.

**`CLAUDE.md`** — one line: `@AGENTS.md`.

**`verify.py`** — Tier 1. Free, no keys, about a minute. It recomputes every
published number from the published data and compares it to what the README
claims. It exits non-zero on any mismatch.

**`.env.example`** — variable names only. Never a value, not even a placeholder
that looks like one.

**`requirements.txt`** — pinned to versions you have actually installed and run,
not guessed.

**`data/`** and **`results/`** — the corpus and the published run.

## The three tiers

Every experiment offers the same three, so a reader can choose how much to trust and
how much to spend.

1. **Verify** — free, no keys, about a minute. Recomputes the published numbers
   from the published data.
2. **Smoke** — paid, a small sample, reports actual spend and projects the full
   run from it.
3. **Full** — paid, the complete experiment.

`AGENTS.md` must run Tier 1 first, and must stop for an explicit yes before Tier 2
or 3. It must explain what a tier will cost and how that estimate was made.

## Rules

**Never commit a secret.** Run the secret scan before you commit. `.env` is
ignored; keep it that way.

**Never commit data you cannot redistribute.** If a experiment uses a third-party
dataset, ship the loader that downloads it and not the dataset. State the licence
in the experiment README.

**Never commit personal data.** No home directory paths, usernames, email
addresses, IP addresses, hostnames or bucket names. Check the manifests: they are
generated and easy to forget.

**Never commit audio.** It is large and it is usually the thing that carries a
licence.

**Run the structural check.** From the repository root:

```bash
python tools/check-experiment.py
```

It asserts every experiment carries the files its paste prompt promises:
`AGENTS.md`, `verify.py`, `METHODOLOGY.md`, `.env.example` and the rest.
It also checks that the prompt's path matches where the experiment
actually lives, and that `.env.example` carries names and no values.
A prompt pointing at a folder that moved is the easiest mistake to make
here and the hardest for a reader to diagnose.

**Verify from a clean clone.** Clone your own repo into a temp directory and run
the tests and `verify.py` there. That is what a stranger gets, and it is the only
way to catch a file you forgot to commit.

## Reporting a difference

If you reproduce a experiment and get different numbers, open an issue with the tier
you ran, the versions from the manifests, and your output. A difference is a
finding, not a complaint. Hosted models change behind fixed version tags; that is
one of the things this repo exists to make visible.
