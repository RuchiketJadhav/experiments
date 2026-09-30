#!/usr/bin/env bash
# Provision a fresh Ubuntu EC2 instance to run tts-bench.
#
# Idempotent: safe to re-run after a reboot or a failed attempt.
#
#   scp -r tts-bench ubuntu@<host>:~/         # or git clone
#   ssh ubuntu@<host> 'bash tts-bench/deploy/bootstrap.sh'
#
# No GPU. Both halves of the run are network-bound HTTP calls to Replicate and
# to the transcription APIs; the instance waits on sockets, it does not
# compute. A GPU instance would cost 20x to do the same waiting.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

echo "== system packages =="
sudo apt-get update -qq
# libsndfile1 is what soundfile binds to; python3-venv is not in the minimal
# Ubuntu AMI. Everything else is already there.
sudo apt-get install -y -qq python3-venv python3-pip libsndfile1

echo "== virtualenv =="
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
. .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r deploy/requirements-cloud.txt

echo "== corpus =="
# Deterministic from the seed plus data/spec/en.yaml, so regenerating on the
# instance and shipping the file both give byte-identical items. Regenerate
# only when it is missing, so a corpus that travelled with the repo is never
# silently replaced by one from a different spec version.
if [ ! -s data/synthetic/en-US.jsonl ]; then
  python gen_corpus.py --locale en-US
else
  echo "   data/synthetic/en-US.jsonl already present, leaving it alone"
fi

echo "== self-test =="
bash run_tests.sh

echo
echo "bootstrap OK."
echo
echo "Next:"
echo "  1. Put the API keys on the instance WITHOUT writing them to a file:"
echo "       export REPLICATE_API_TOKEN=... OPENAI_API_KEY=... DEEPGRAM_API_KEY=..."
echo "     or fetch them at launch from SSM Parameter Store / Secrets Manager:"
echo "       export OPENAI_API_KEY=\$(aws ssm get-parameter --name /tts-bench/openai \\"
echo "                                  --with-decryption --query Parameter.Value --output text)"
echo "  2. bash deploy/run_bench.sh configs/synthetic_en.yaml runs/syn-en"
