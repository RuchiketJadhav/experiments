# Runbook: <study name>

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

<Minimum Python version. Any system library, and the install command per OS. Any
tool that is deliberately NOT needed, so nobody installs it.>

```bash
python --version
python -m venv .venv
. .venv/bin/activate                 # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Report the OS, the Python version and whether the install succeeded.

## Step 2: tests

```bash
bash run_tests.sh
```

All suites should pass with no network and no API key. If any fail, stop and show
the output.

## Step 3: Tier 1 — Verify (free, always run this first)

```bash
python verify.py
```

**Show the user the full table.** Then say plainly what it proves and what it does
not: it proves the published numbers were computed from the published data, not
that the systems behave that way today.

If anything fails, stop and show which check failed and by how much.

**For most people this is the whole job.** Do not push them toward a paid tier.

## Step 4: keys, only if they want a paid tier

**These are the reader's own API keys, paying from the reader's own accounts.**
Nothing in this repository carries a key, and nothing here can call a provider on
our behalf. If a key is missing the run fails at startup rather than part way
through, because `expand_env()` in `adapters/httputil.py` raises on an unset
variable instead of sending an empty header.

```bash
cp .env.example .env
```

Tell the user to open `.env` and fill in the values **themselves**. Never ask
them to paste a key into the chat, and never print one back.

Then load the file into the environment. This has to happen BEFORE the check below,
or the check reports NOT SET for keys that are perfectly well filled in:

```bash
set -a; . ./.env; set +a          # bash / zsh / git-bash
```

```powershell
Get-Content .env | Where-Object { $_ -match '^\s*[^#].*=' } | ForEach-Object {
    $name, $value = $_ -split '=', 2
    [Environment]::SetEnvironmentVariable($name.Trim(), $value.Trim(), 'Process')
}
```

Now verify presence only, never values:

```bash
python -c "
import os
for k in (<the variable names>):
    print(f'{k}: {\"set\" if os.environ.get(k) else \"NOT SET\"}')
"
```

Every variable should print `set`. If any says `NOT SET`, say which one, and stop. Do not
guess at a value and do not continue to a paid tier with a missing key.

## Step 5: Tier 2 — Smoke (paid; ask first)

**Before running, tell the user the size, the cost, HOW you derived the cost, and
the expected time. Then wait for a yes.**

<The commands. Say which sampling flag to use and why the obvious alternative is
wrong.>

Afterwards report the real spend and the real elapsed time, then project the full
run from those measured numbers rather than from the estimate.

## Step 6: Tier 3 — Full (paid; ask first)

**Before running, tell the user the size, the cost and the time. Then wait for a
yes.**

<The commands, including the sharded or distributed path if there is one. Any
operational trap worth passing on.>

## Step 7: compare honestly

Compare against the published results and say plainly what matches and what does
not.

<The expected size of difference, and why. Sampling noise at this n. Anything that
can change underneath a pinned version.>

- **Within <X>**: consistent with the study. Say so.
- **More than <Y>, or a change in ordering**: a real difference. Report it as a
  finding, not an error.

Never adjust, round or explain away a difference to make it match. The point of
the repo is that someone can disagree with us using our own tools.

## If something breaks

<Known failure modes and their fixes. Provider errors, rate limits, resumability.>
