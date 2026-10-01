# Dograh Experiments

Voice AI experiments from the Dograh team, with data you can check.

Each experiment ships its data, its code and its results together, so you can
recompute every number we publish and disagree with us using our own tools.

Read them at **[dograh.com/hub/experiments](https://dograh.com/hub/experiments)**.
This repository is the proof behind them: no website code lives here.

## Experiments

| experiment | tag | question | headline finding | page |
|---|---|---|---|---|
| [tts-verbalization](benchmarks/tts-verbalization/) | Benchmark | Do open text-to-speech models say dates, prices and phone numbers correctly? | The smallest of seven models is the most accurate and uses roughly 18x less compute. No model can say a hashtag. Best date score 0.23. | [read](https://dograh.com/hub/experiments/tts-verbalization-benchmark) |

**Benchmark** compares models. **Test** examines one thing in depth. They live in
`benchmarks/` and `tests/` respectively.

## Reproduce any experiment with one prompt

Every experiment is built so an AI agent can take it from a clone to our numbers
with no other context. Open your agent (Claude Code or Cursor) and paste this,
swapping in the experiment you want:

> Reproduce Dograh's TTS verbalization experiment: run `git clone --depth 1 https://github.com/dograh-hq/experiments.git && cd experiments/benchmarks/tts-verbalization`, then follow AGENTS.md. Start with the free verification, and ask me before spending anything.

Each experiment folder holds:

- **`README.md`** — the headline finding and results up top, then the full write-up
- **`METHODOLOGY.md`** — how it was run, in enough detail to argue with
- **`AGENTS.md`** — the runbook your agent follows
- **`verify.py`** — Tier 1, free, no keys, about a minute

## The three tiers

Every experiment offers the same three, so you can choose how much to trust and
how much to spend.

**Tier 1, Verify.** Free. No API keys. About a minute. Recomputes every published
number from the published data with the experiment's own scoring code and checks it
against what the README claims. This is what most people should run: it proves the
numbers were computed honestly from the data in this repo. It does not prove
anything about how the models behave today.

**Tier 2, Smoke.** Paid. A small sample through the real models, reporting what it
actually cost and projecting the full run from that.

**Tier 3, Full.** Paid. The complete experiment. Expect hours and real money; each
experiment says how much.

An agent following `AGENTS.md` always runs Tier 1 first, and always stops for your
explicit approval before spending anything.

## Why the data ships with the results

An earlier version of the TTS experiment used a corpus licensed for use but not for
redistribution. We could publish results, but nobody could check them, which makes a
benchmark unfalsifiable by anyone except its author. Everything here is generated or
owned, so the data, the generator and the results publish together.

Where an experiment's code can fetch a third-party dataset at run time, that code
ships and the dataset does not.

## Adding an experiment

Copy `_template/` into `benchmarks/` or `tests/` and fill it in. See
[CONTRIBUTING.md](CONTRIBUTING.md).

## Licence

Code is Apache-2.0 ([LICENSE](LICENSE)).

Data and results, meaning everything under `benchmarks/*/data/`,
`benchmarks/*/results/` and the same paths under `tests/`, are CC BY 4.0
([LICENSE-DATA](LICENSE-DATA)). Use them for anything, including commercially;
credit Dograh.

To cite: see [CITATION.cff](CITATION.cff), or use GitHub's **Cite this repository**
button.

## Who made this

[Dograh](https://dograh.com), an open source voice AI builder. We publish the
experiments we run to decide our own engineering, because a benchmark nobody can
check is marketing.
