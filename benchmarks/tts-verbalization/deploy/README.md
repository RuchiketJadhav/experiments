# Running tts-bench on an AWS instance

The benchmark is 15,120 HTTP calls that mostly wait. Nothing here computes:
the voices run on Replicate, the recognizers run on OpenAI and Deepgram. The
instance is a place to wait from, which is the whole reason to move it off a
laptop -- the previous full run took six hours and died twice because the
machine slept.

## What to launch

| | |
|---|---|
| AMI | Ubuntu 24.04 LTS (x86_64) |
| Instance | `t3.small` (2 vCPU, 2 GiB) |
| Root volume | 30 GiB gp3 |
| GPU | **no** |
| Security group | outbound 443 only; no inbound but SSH from your address |

`t3.small` is not a guess. Peak memory is one WAV in flight per worker plus the
JSON of the run; the 5.8 GB laptop ceiling that killed the two-lineage run came
from running a local recognizer, which the API path does not do. The 30 GiB
volume is for the audio: seven systems x 1,080 items is roughly 1 GB of WAV,
and the rest is headroom so a full disk never truncates `results.jsonl`.

A GPU instance costs roughly 20x to perform the same waiting.

## Do not expose this to a network

`adapters/tts/http.py` and `adapters/httputil.py` take arbitrary URLs from the
config file and fetch them. That is a server-side request forgery primitive:
anyone who can influence a config can make the instance fetch
`http://169.254.169.254/` and read the instance metadata. It is harmless as a
local tool and dangerous as a service. **Do not run `app.py` on a public
address.** If a hosted UI is ever wanted, it needs a scheme/host allowlist and
private-range blocking first, plus IMDSv2 enforced on the instance.

Keep the security group outbound-only and put the instance in a private subnet
with a NAT gateway if you have one.

## Secrets

Keys go in the environment, never in the image and never in a file on the
instance.

```bash
export REPLICATE_API_TOKEN=$(aws ssm get-parameter --name /tts-bench/replicate \
        --with-decryption --query Parameter.Value --output text)
export OPENAI_API_KEY=$(aws ssm get-parameter --name /tts-bench/openai \
        --with-decryption --query Parameter.Value --output text)
export DEEPGRAM_API_KEY=$(aws ssm get-parameter --name /tts-bench/deepgram \
        --with-decryption --query Parameter.Value --output text)
```

Attach an instance role with `ssm:GetParameter` on `/tts-bench/*` and
`kms:Decrypt` on the key that encrypts them, plus `s3:PutObject` on the results
bucket. Nothing else.

The run manifest is a deliberate field allowlist and there is a test asserting
no `authorization` or `bearer` value reaches it, so `manifest.json` is safe to
copy off the box.

## Sequence

```bash
# on your machine
rsync -av --exclude .venv --exclude runs --exclude data/raw \
      tts-bench/ ubuntu@$HOST:~/tts-bench/

# on the instance
bash tts-bench/deploy/bootstrap.sh          # deps, venv, corpus, self-test
cd tts-bench
export REPLICATE_API_TOKEN=... OPENAI_API_KEY=... DEEPGRAM_API_KEY=...

# 20-item smoke first: confirms cost and latency before committing to 15,120 calls
python run.py --config configs/synthetic_en.yaml --out runs/smoke --sample 20
python score.py --run runs/smoke --asr whisper

# the full run, watchdogged, under tmux so an SSH drop does not kill it
tmux new -s bench
S3_BUCKET=my-results-bucket bash deploy/run_bench.sh configs/synthetic_en.yaml runs/syn-en
```

`run_bench.sh` computes the expected record count from the config and arms
`watchdog.sh` with it. The watchdog restarts the runner if `results.jsonl`
stops growing for seven minutes; resume is free because audio and transcripts
are content-addressed and cached, so a needless restart costs one item.

## What it costs and how long it takes

1,080 items x 7 systems = 7,560 syntheses, then 15,120 transcriptions
(two recognizers).

- **Replicate** bills official models per input character and community models
  per hardware-second including cold boot. The 540-item run came to a few
  dollars per model; this corpus is twice the size with longer carrier
  sentences, so budget accordingly and check the smoke run's actual spend
  before launching the full one.
- **Instance**: `t3.small` on-demand is about $0.02/hour. Even a 20-hour run is
  under a dollar. The instance is the cheapest part by a wide margin; do not
  optimise it.
- **Wall clock**: `runs/open8/manifest.json` is the only honest source here --
  3,754 new records in 6.03 hours, i.e. **10.4 records/minute sustained**.
  That puts 15,120 records at roughly **24 hours serially**. (An earlier note
  in this file said 90 hours, from a 2.8/min figure taken across a stalled
  stretch rather than a healthy one. 10.4/min is the real rate.)

  Do not attack that with a bigger instance -- the time is spent waiting on
  Replicate, not computing. Until `--workers` lands, shard by TTS system:
  one process per model, each with its own `--out` directory, then merge.
  Audio filenames are `{item_id}__{hash(tts_id, revision, text, sample)}`,
  so two systems can never collide, and `score.py` reads only
  `results.jsonl` -- it needs no manifest. `deploy/shard_by_model.sh` does
  this; expect roughly the slowest single model's time, about 4-6 hours.

## Getting the results back

`run_bench.sh` syncs to `$S3_BUCKET` if it is set, excluding `audio/` unless
`SYNC_AUDIO=1`. What you need to analyse a run is `results.jsonl`,
`manifest.json` and `report.html`; the audio only matters for a listening
audit.

Unlike every previous run, this one's `results.jsonl` **can** be published: the
corpus is generated by `gen_corpus.py` from `data/spec/en.yaml` and a seed, so
no third-party licence attaches to the rows. `score.py --no-evidence` exists
for the PolyNorm runs and is not needed here.
