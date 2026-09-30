#!/usr/bin/env bash
# Run every suite and FAIL if any of them fails.
#
# The README told you to run `for t in tests/*.py; do python "$t"; done`, which
# reports a red suite and then exits 0, so CI and the bootstrap script would
# both call a broken checkout healthy. This propagates the status.
cd "$(dirname "$0")" || exit 1
# Pick an interpreter that can actually import the dependencies. Naming one is
# not enough: on Windows `python3` resolves to an App Execution Alias that
# starts a bare interpreter with none of the project's packages, so the suites
# failed on `import yaml` while the same tests passed under `python`.
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for candidate in python3 python py; do
    if command -v "$candidate" >/dev/null 2>&1 &&
       "$candidate" -c "import yaml, numpy" >/dev/null 2>&1; then
      PY="$candidate"; break
    fi
  done
fi
if [ -z "$PY" ]; then
  echo "no python with the project dependencies found; set PYTHON=..." >&2
  exit 1
fi
echo "using $PY ($("$PY" --version 2>&1))"

failed=()
for t in tests/*.py; do
  if PYTHONIOENCODING=utf-8 "$PY" "$t" > /tmp/tts-bench-test.$$ 2>&1; then
    tail -n 1 /tmp/tts-bench-test.$$
  else
    failed+=("$t")
    echo "FAIL $t"
    cat /tmp/tts-bench-test.$$
  fi
done
rm -f /tmp/tts-bench-test.$$

if [ ${#failed[@]} -ne 0 ]; then
  echo
  echo "${#failed[@]} suite(s) failed: ${failed[*]}"
  exit 1
fi
echo
echo "all suites passed"
