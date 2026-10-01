# <Study title: the question, as a question>

<One paragraph: what was measured, on what, with what. Then the sentence that
makes this page trustworthy: every number here is reproduced from the data in
`results/` by `verify.py`, which takes about a minute and costs nothing.>

**Reproduce this study.** Paste into Claude Code:

> Reproduce Dograh's <experiment> experiment: run `git clone --depth 1 https://github.com/dograh-hq/experiments.git && cd experiments/benchmarks/<experiment>`, then follow AGENTS.md. Start with the free verification, and ask me before spending anything.

---

## The question

<What you measured, in plain language, with a concrete example of a pass and a
fail. Then why it matters to someone building something. Not the method yet.>

## Headline findings

<Three to five bold claims, each with the number that supports it. If a finding
is counterintuitive, say so and say why it happens. No hedging, no throat
clearing.>

## Results

<The main table, inside a verify block so verify.py checks it. Say what the
metric means and which measurement is primary.>

<!-- verify:results -->
| | | |
|---|---|---|
<!-- /verify:results -->

<Call out any statistical ties explicitly. A reader will otherwise read an
ordering into two numbers that are the same.>

## <Breakdown table: per category, per condition, whatever the second axis is>

<!-- verify:categories -->
| | | |
|---|---|---|
<!-- /verify:categories -->

<The point the breakdown makes that the headline hides.>

## <A section per finding that needs its mechanism explained>

<Do not just report that something scored badly. Say what is actually happening,
with evidence from the data. If you did not verify a mechanism, do not claim one.>

## The data

<!-- verify:facts -->
- `unique_records` = ``
- `items` = ``
<!-- /verify:facts -->

<Where the corpus came from, how it was built, what makes it trustworthy, and how
to regenerate it. If any of it is a judgement call, say which and why.>

## Limitations

<Bullets. What this does not measure, what it cannot support, what is a
convention of ours rather than a fact, and what was lost or excluded and why. Be
harder on yourself than a reader would be.>

## Reproduce

### Tier 1 — Verify (free, no keys, about a minute)

```bash
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python verify.py
```

<What it proves, and what it does not.>

### Tier 2 — Smoke (paid, needs API keys)

<Size, what it costs, how long.>

### Tier 3 — Full (paid)

<Size, what it costs, how long, and what a fresh run should be expected to differ
by.>

## Files

```
<the tree, annotated one line each>
```

## Licence

Code BSD-2-Clause (`../../LICENSE`). Data and results CC BY 4.0
(`../../LICENSE-DATA`). <Any third-party dataset, its licence, and a plain
statement that no row of it appears in this repository.>
