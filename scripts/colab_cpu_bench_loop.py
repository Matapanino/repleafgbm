#!/usr/bin/env python3
"""Keep-alive Colab CPU driver for the fair leaderboard (resumable, self-healing).

Colab CPU recycles the VM on long ``colab exec`` runs (a ~70-min run loses its
results before download; a ~36-min run survives). So we keep ONE VM alive, build
the Rust extension **once** on it, and run the leaderboard in short, durable
**per-(dataset, seed) slices** (5 cells, ~10-15 min each), downloading the ledger
to a local file after every slice so progress is never lost. If the VM dies, we
re-provision, re-upload the accumulated ledger, rebuild Rust, and continue —
the ledger's ``(suite, dataset, model, seed)`` keys make every retry idempotent.

This orchestrator runs locally and drives the Colab CLI; the per-slice work runs
via ``scripts/colab_remote_bench.py`` (which builds Rust once per VM and resumes
from the uploaded ledger). Re-run this script any time to continue an unfinished
suite.

Usage::

    PYTHONPATH=src python3 scripts/colab_cpu_bench_loop.py \\
        --suite grinsztajn_num_reg --seeds 5 --n-trials 50 [--max-slices N]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DRIVER = "scripts/colab_remote_bench.py"
EXEC_TIMEOUT = "1800"  # idle timeout per slice exec


def colab(*args: str, check: bool = False) -> subprocess.CompletedProcess:
    print("  $ colab", *args, flush=True)
    return subprocess.run(["colab", *args], cwd=ROOT, text=True, check=check)


def session_alive(name: str) -> bool:
    # `colab status` exits 0 even for a missing session, so check the active list.
    r = subprocess.run(["colab", "sessions"], cwd=ROOT,
                       capture_output=True, text=True)
    return name in r.stdout


def suite_datasets(suite: str) -> list[str]:
    from benchmarks.suites import get_suite
    return [d.name for d in get_suite(suite).datasets]


def families() -> list[str]:
    from benchmarks.hpo import FAMILIES
    return list(FAMILIES)


def pending_slices(suite: str, datasets: list[str], seeds: list[int],
                   ledger_path: Path, fams: list[str]) -> list[tuple[str, int]]:
    """(dataset, seed) pairs not yet complete for every family in the ledger."""
    done: set[tuple[str, str, int]] = set()
    if ledger_path.exists():
        for line in ledger_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "key" in o:
                done.add((o["dataset"], o["model"], int(o["seed"])))
    out = []
    for ds in datasets:
        for s in seeds:
            if not all((ds, m, s) in done for m in fams):
                out.append((ds, s))
    return out


def provision(name: str, tarball: Path, ledger_path: Path) -> bool:
    """Fresh VM: upload the working tree and (if present) the accumulated ledger
    so the first slice's driver restores it before computing. Returns success."""
    print(f">> provisioning CPU VM '{name}'", flush=True)
    if colab("new", "-s", name).returncode != 0:
        return False
    if colab("upload", "-s", name, str(tarball),
             "/content/rlgbm.tar.gz").returncode != 0:
        return False
    if ledger_path.exists():
        if colab("upload", "-s", name, str(ledger_path),
                 "/content/ledger_in.jsonl").returncode != 0:
            return False
    return True


def run_slice(name: str, suite: str, ds: str, seed: int, trials: int,
              extra: list[str], ledger_path: Path) -> bool:
    """One durable slice: upload argv, exec the driver, download the ledger.

    Returns True iff the exec AND the ledger download both succeed (progress
    locked locally)."""
    argv = ["--suite", suite, "--seed-list", str(seed), "--datasets", ds,
            "--n-trials", str(trials), "--out", "/content/leaderboard.md",
            "--ledger", "/content/ledger.jsonl", *extra]
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("\n".join(argv) + "\n")
        argv_file = f.name
    # check=False everywhere: a failed call means the VM died -> return False so
    # the loop re-provisions (rather than crashing the whole run).
    if colab("upload", "-s", name, argv_file,
             "/content/bench_argv.txt").returncode != 0:
        return False
    if colab("exec", "-s", name, "--timeout", EXEC_TIMEOUT,
             "-f", DRIVER).returncode != 0:
        return False
    # Pull the accumulated ledger back so this slice's cells are durable locally.
    return colab("download", "-s", name, "/content/ledger.jsonl",
                 str(ledger_path)).returncode == 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", required=True)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--n-trials", type=int, default=50)
    p.add_argument("--session", default="rlgbm-cpu-loop")
    p.add_argument("--max-slices", type=int, default=0,
                   help="stop after N slices this invocation (0 = until done)")
    p.add_argument("--max-fails", type=int, default=5,
                   help="abort after this many consecutive slice failures")
    p.add_argument("extra", nargs="*", help="extra args passed to leaderboard.py")
    args = p.parse_args(argv)

    ledger_path = ROOT / "benchmarks" / "results" / f"leaderboard_{args.suite}.jsonl"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    datasets = suite_datasets(args.suite)
    fams = families()
    seeds = list(range(args.seeds))

    # Snapshot the working tree once; re-uploaded on every (re)provision.
    tarball = Path(tempfile.mktemp(suffix=".tar.gz", prefix="rlgbm-"))
    subprocess.run(
        ["tar", "--exclude=.git", "--exclude=*/__pycache__", "--exclude=*.egg-info",
         "--exclude=target", "--exclude=build", "--exclude=dist",
         "--exclude=.pytest_cache", "-czf", str(tarball), "."],
        cwd=ROOT, check=True)

    done_count = 0
    fails = 0
    try:
        while True:
            todo = pending_slices(args.suite, datasets, seeds, ledger_path, fams)
            if not todo:
                print(f">> SUITE COMPLETE: {args.suite} ({args.seeds} seeds, "
                      f"{len(datasets)} datasets) — nothing pending", flush=True)
                break
            if args.max_slices and done_count >= args.max_slices:
                print(f">> reached --max-slices={args.max_slices}; "
                      f"{len(todo)} slices still pending (re-run to continue)",
                      flush=True)
                break
            if fails >= args.max_fails:
                print(f">> ABORT: {fails} consecutive failures; "
                      f"{len(todo)} slices pending. Re-run to retry.", flush=True)
                return 1

            if not session_alive(args.session):
                if not provision(args.session, tarball, ledger_path):
                    fails += 1
                    print(f"  [provision FAILED] (fail {fails}/{args.max_fails})",
                          flush=True)
                    colab("stop", "-s", args.session)
                    time.sleep(5)
                    continue

            ds, seed = todo[0]
            print(f">> slice {done_count + 1}: {ds} seed={seed} "
                  f"({len(todo)} pending)", flush=True)
            if run_slice(args.session, args.suite, ds, seed, args.n_trials,
                         args.extra, ledger_path):
                done_count += 1
                fails = 0
                print(f"  [slice ok] {ds} seed={seed}", flush=True)
            else:
                fails += 1
                print(f"  [slice FAILED] {ds} seed={seed} — VM likely recycled; "
                      f"re-provisioning (fail {fails}/{args.max_fails})", flush=True)
                colab("stop", "-s", args.session)  # force fresh VM next iteration
                time.sleep(5)
    finally:
        colab("stop", "-s", args.session)
        tarball.unlink(missing_ok=True)

    remaining = len(pending_slices(args.suite, datasets, seeds, ledger_path, fams))
    print(f">> done this run: {done_count} slices; {remaining} still pending; "
          f"ledger: {ledger_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
