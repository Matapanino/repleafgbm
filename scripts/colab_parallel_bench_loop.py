#!/usr/bin/env python3
"""Local keep-alive orchestrator for the parallel Colab leaderboard run.

Drives ``scripts/colab_parallel_bench.py`` on ONE many-core Colab session
(e.g. ``--runtime tpu-v5e1``: 24-vCPU AMD EPYC host) to run the fair
leaderboard's full (suite × dataset × seed) grid in parallel waves, keeping
the Mac free. Progress is durable on both sides: the VM appends every worker
ledger into per-suite masters, and this loop downloads the masters after every
wave into ``--run-dir``. If the VM dies, it re-provisions, re-uploads the
tarball + the local masters, re-runs setup (deps, Rust build, sequential
dataset prefetch), and continues — ledger keys make every retry idempotent.

Provenance: the tarball is ``git archive <sha>`` (committed tree only), so the
run is clean-by-construction; the pinned SHA is recorded in the run-dir
manifest (the VM has no .git, so the in-ledger git fields are null there).

Usage (from the repo root)::

    PYTHONPATH=src nohup python3 scripts/colab_parallel_bench_loop.py \
        --sha <commit> --session rlgbm-tpu > colab_lb.log 2>&1 &
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

DRIVER = str(ROOT / "scripts" / "colab_parallel_bench.py")
SUITES = ["grinsztajn_num_reg", "grinsztajn_num_cls",
          "grinsztajn_cat_reg", "grinsztajn_cat_cls"]
FAMILIES = ["repleaf", "lightgbm", "xgboost", "catboost",
            "hist_gradient_boosting"]


def colab(*args: str, timeout: int | None = None) -> int:
    """Run a colab CLI command with a HARD wall-clock timeout.

    The CLI's own ``--timeout`` is an *idle* timeout that does not fire on a
    dead connection — a wave exec once hung for 8 hours. ``subprocess.run``
    kills the child on expiry, converting hangs into a bounded loss the
    re-provision path absorbs.
    """
    print("  $ colab", *args, flush=True)
    try:
        return subprocess.run(["colab", *args], cwd=ROOT, text=True,
                              timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        print(f"  colab call exceeded hard timeout ({timeout}s); killed",
              flush=True)
        return 124


def session_alive(name: str) -> bool:
    try:
        r = subprocess.run(["colab", "sessions"], cwd=ROOT,
                           capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return False
    return name in r.stdout


def all_tasks(seeds: int) -> list[dict]:
    from benchmarks.suites import get_suite
    tasks = []
    for suite in SUITES:
        for ds in get_suite(suite).datasets:
            for seed in range(seeds):
                tasks.append({"suite": suite, "dataset": ds.name, "seed": seed})
    return tasks


def done_keys(run_dir: Path) -> set[str]:
    keys: set[str] = set()
    for suite in SUITES:
        path = run_dir / f"{suite}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("key"):
                keys.add(obj["key"])
    return keys


def remaining(tasks: list[dict], keys: set[str]) -> list[dict]:
    from benchmarks.ledger import cell_key
    out = []
    for t in tasks:
        if not all(cell_key(t["suite"], t["dataset"], fam, t["seed"]) in keys
                   for fam in FAMILIES):
            out.append(t)
    return out


def upload_json(session: str, obj: dict, remote: str) -> bool:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(obj, fh)
        tmp = fh.name
    return colab("upload", "-s", session, tmp, remote, timeout=300) == 0


def provision(session: str, args, tarball: Path, run_dir: Path) -> bool:
    """(Re)create the session and run setup; re-seed VM masters from local.

    A dead VM can linger in ``colab sessions`` as a zombie (uploads then fail
    with remote not-found), so a failed first upload forces a stop + fresh
    session before retrying once.
    """
    def create() -> bool:
        runtime = args.runtimes[args.runtime_idx % len(args.runtimes)]
        new_cmd = ["new", "-s", session]
        if runtime.startswith("tpu-"):
            new_cmd += ["--tpu", runtime.removeprefix("tpu-")]
        elif runtime != "cpu":
            new_cmd += ["--gpu", runtime]
        print(f"provisioning runtime={runtime}", flush=True)
        return colab(*new_cmd, timeout=600) == 0

    if not session_alive(session) and not create():
        return False
    if colab("upload", "-s", session, str(tarball),
             "/content/rlgbm.tar.gz", timeout=300) != 0:
        print("upload failed on a listed session -> zombie; recreating",
              flush=True)
        colab("stop", "-s", session)
        if not create():
            return False
        if colab("upload", "-s", session, str(tarball),
                 "/content/rlgbm.tar.gz", timeout=300) != 0:
            return False
    if not upload_json(session, {"mode": "setup", "suites": SUITES},
                       "/content/wave_spec.json"):
        return False
    if colab("exec", "-s", session, "--timeout", str(args.exec_timeout),
             "-f", DRIVER, timeout=2400) != 0:  # setup: deps+rust+prefetch
        return False
    ok = True
    for suite in SUITES:
        local = run_dir / f"{suite}.jsonl"
        if local.exists() and local.stat().st_size:
            ok &= colab("upload", "-s", session, str(local),
                        f"/content/ledgers/{suite}.jsonl", timeout=300) == 0
    return ok


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sha", required=True,
                   help="commit to run (git archive; clean by construction)")
    p.add_argument("--session", default="rlgbm-tpu")
    p.add_argument("--runtime", dest="runtime_csv", default="tpu-v5e1,tpu-v6e1",
                   help="comma-separated fallback chain: tpu-v5e1|tpu-v6e1|cpu"
                        "|T4/L4/... (GPU); rotates after repeated provision "
                        "failures (pool capacity comes and goes)")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--n-trials", type=int, default=50)
    p.add_argument("--max-rows", type=int, default=20000)
    p.add_argument("--workers", type=int, default=20)
    p.add_argument("--tasks-per-wave", type=int, default=20)
    p.add_argument("--exec-timeout", default="2700")
    p.add_argument("--run-dir", type=Path,
                   default=ROOT / "benchmarks" / "results" / "colab10seed")
    p.add_argument("--max-waves", type=int, default=1000)
    args = p.parse_args(argv)
    args.runtimes = [r.strip() for r in args.runtime_csv.split(",") if r.strip()]
    args.runtime_idx = 0

    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps({
        "sha": args.sha, "runtime": args.runtime_csv, "seeds": args.seeds,
        "n_trials": args.n_trials, "max_rows": args.max_rows,
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": "tarball = git archive <sha>: committed tree only, "
                "clean by construction",
    }, indent=2))

    tarball = run_dir / f"rlgbm-{args.sha}.tar.gz"
    if not tarball.exists():
        with open(tarball, "wb") as fh:
            subprocess.run(["git", "archive", "--format=tar.gz", args.sha],
                           cwd=ROOT, stdout=fh, check=True)

    tasks = all_tasks(args.seeds)
    need_setup = True
    provision_fails = 0
    for wave_no in range(args.max_waves):
        todo = remaining(tasks, done_keys(run_dir))
        if not todo:
            print("ALL CELLS DONE", flush=True)
            return 0
        print(f"=== wave {wave_no}: {len(todo)} tasks remaining ===",
              flush=True)
        if need_setup:
            if not provision(args.session, args, tarball, run_dir):
                provision_fails += 1
                if provision_fails % 3 == 0:
                    args.runtime_idx += 1  # pool capacity comes and goes
                    print("rotating runtime ->",
                          args.runtimes[args.runtime_idx % len(args.runtimes)],
                          flush=True)
                print("provision failed; retrying in 60s", flush=True)
                time.sleep(60)
                continue
            need_setup = False
            provision_fails = 0
        batch = todo[: args.tasks_per_wave]
        spec = {"mode": "wave", "suites": SUITES, "tasks": batch,
                "workers": args.workers, "n_trials": args.n_trials,
                "max_rows": args.max_rows}
        if not upload_json(args.session, spec, "/content/wave_spec.json"):
            need_setup = True
            continue
        if colab("exec", "-s", args.session, "--timeout",
                 str(args.exec_timeout), "-f", DRIVER, timeout=3300) != 0:
            print("exec failed -> assuming VM died; re-provisioning",
                  flush=True)
            need_setup = True
            continue
        got_any = False
        for suite in SUITES:
            rc = colab("download", "-s", args.session,
                       f"/content/ledgers/{suite}.jsonl",
                       str(run_dir / f"{suite}.jsonl"), timeout=300)
            got_any |= rc == 0
        if not got_any:
            need_setup = True
    print("max waves reached", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
