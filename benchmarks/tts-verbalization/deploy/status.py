#!/usr/bin/env python3
"""One-screen answer to "is it progressing, and how much is left?"

    python3 deploy/status.py [run-dir]
    python3 deploy/status.py [run-dir] --watch     # logger; run under tmux

Record COUNT is not progress. results.jsonl is append-only, so a retried item
leaves the old failure in the file and adds a new record: a shard can grow by a
thousand lines while succeeding at nothing. What matters is, per unique key,
what the LATEST record says -- which is also exactly how score.py reads it.
That distinction is not academic here: a run that looked complete at 15,120
records had a third of its syntheses failed.

--watch appends a timestamped line to <run-dir>/progress.tsv every 5 minutes,
so rate and ETA are measured rather than guessed.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

TARGET_PER_SHARD = 2160          # 1,080 items x 2 recognizers
RETRYABLE = {"synthesis_error", "asr_error"}


def snapshot(root: Path) -> dict:
    rows, tot_done, tot_left = [], 0, 0
    for d in sorted((root / "shards").iterdir() if (root / "shards").exists() else []):
        f = d / "results.jsonl"
        if not f.is_file():
            continue
        latest: dict[str, str] = {}
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue                  # truncated final line: expected
                latest[r["key"]] = r["status"]
        done = sum(1 for s in latest.values() if s == "scored")
        left = sum(1 for s in latest.values() if s in RETRYABLE)
        rows.append({"shard": d.name, "scored": done, "to_retry": left,
                     "other": len(latest) - done - left,
                     "mtime": f.stat().st_mtime})
        tot_done += done
        tot_left += left
    return {"rows": rows, "scored": tot_done, "to_retry": tot_left}


def report(root: Path) -> None:
    snap = snapshot(root)
    rows = snap["rows"]
    if not rows:
        print(f"no shards under {root}/shards")
        return
    now = time.time()
    print(f"{'shard':<26}{'scored':>8}{'to retry':>10}{'other':>7}"
          f"{'done':>7}{'last write':>12}")
    print("-" * 70)
    for r in rows:
        age = now - r["mtime"]
        stamp = f"{age/60:.0f}m ago" if age < 5400 else f"{age/3600:.1f}h ago"
        pct = 100.0 * r["scored"] / TARGET_PER_SHARD
        print(f"{r['shard']:<26}{r['scored']:>8}{r['to_retry']:>10}"
              f"{r['other']:>7}{pct:>6.0f}%{stamp:>12}")
    print("-" * 70)
    total_target = len(rows) * TARGET_PER_SHARD
    print(f"{'TOTAL':<26}{snap['scored']:>8}{snap['to_retry']:>10}{'':>7}"
          f"{100*snap['scored']/total_target:>6.0f}%")

    p = root / "progress.tsv"
    if not p.exists():
        print("\n(no progress.tsv yet -- start the logger for rate and ETA:"
              "\n   tmux new -d -s watch 'python3 deploy/status.py"
              f" {root} --watch')")
        return
    lines = [l.split("\t") for l in p.read_text().splitlines()[1:] if l.strip()]
    if len(lines) < 2:
        print("\n(logger running; need two samples for a rate)")
        return
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    t0 = time.mktime(time.strptime(lines[0][0], fmt))
    t1 = time.mktime(time.strptime(lines[-1][0], fmt))
    d0, d1 = int(lines[0][1]), int(lines[-1][1])
    hours = (t1 - t0) / 3600
    if hours <= 0:
        return
    if d1 <= d0:
        print(f"\nNO PROGRESS in the last {hours:.1f}h -- check the workers")
        return
    rate = (d1 - d0) / (hours * 60)
    eta = snap["to_retry"] / rate / 60 if rate else float("inf")
    print(f"\nrate {rate:.1f} scored/min over {hours:.1f}h"
          f"  ->  ETA {eta:.1f}h for the {snap['to_retry']} still to retry")


def watch(root: Path) -> None:
    out = root / "progress.tsv"
    if not out.exists() or not out.stat().st_size:
        out.write_text("timestamp\tscored\tto_retry\n", encoding="utf-8")
    while True:
        snap = snapshot(root)
        with out.open("a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\t"
                     f"{snap['scored']}\t{snap['to_retry']}\n")
        time.sleep(300)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--watch"]
    target = Path(args[0]) if args else Path("runs/syn-en")
    if "--watch" in sys.argv[1:]:
        watch(target)
    else:
        report(target)
