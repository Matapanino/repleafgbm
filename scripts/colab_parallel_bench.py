"""On-VM wave driver for the fair leaderboard on a many-core Colab host.

Companion to ``scripts/colab_parallel_bench_loop.py`` (which runs locally and
drives the Colab CLI). Unlike the sequential ``colab_remote_bench.py`` slice
driver, this one exploits a many-vCPU host (e.g. the 24-vCPU AMD EPYC behind a
TPU v5e1 runtime) by running one leaderboard cell-slice per worker process,
``OMP_NUM_THREADS=1`` each, in parallel.

Protocol (files under ``/content``):

  * ``rlgbm.tar.gz``      — ``git archive`` of the pinned commit (upload once).
  * ``wave_spec.json``    — {"mode": "setup"|"wave", ...} uploaded per exec.
  * ``ledgers/<suite>.jsonl`` — per-suite master ledgers; the wave appends each
    worker's private ledger into its suite master, and the local loop downloads
    the masters after every wave, so progress is durable on both sides.

``mode: setup`` extracts the repo, installs the [bench] deps, builds the Rust
extension, and *sequentially prefetches every dataset* so parallel workers
never race the OpenML cache. ``mode: wave`` runs ``spec["tasks"]`` — each
``{suite, dataset, seed}`` is one ``benchmarks.leaderboard`` invocation
(5 model-cells) — under a thread pool of ``spec["workers"]`` subprocesses.
Per-task completion lines keep the exec's idle timeout at bay.
"""

import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = "/content/repleafgbm"
TARBALL = "/content/rlgbm.tar.gz"
SPEC = "/content/wave_spec.json"
LEDGER_DIR = "/content/ledgers"
SETUP_MARKER = "/content/.rlgbm_parallel_setup_done"
ENV = {**os.environ, "OMP_NUM_THREADS": "1", "PYTHONPATH": f"{REPO}/src"}


def _run(cmd, **kw):
    print("+", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, **kw)


def setup(spec):
    os.makedirs(REPO, exist_ok=True)
    with tarfile.open(TARBALL, "r:gz") as tf:
        try:
            tf.extractall(REPO, filter="data")
        except TypeError:
            tf.extractall(REPO)
    print(f"extracted {TARBALL} -> {REPO}", flush=True)
    _run([sys.executable, "-m", "pip", "install", "-q",
          "optuna>=3", "catboost>=1.2", "xgboost>=2.0", "lightgbm>=4",
          "matplotlib>=3.5"], check=False)
    # Rust backend: RepLeaf dominates HPO cost on NumPy; bitwise-identical.
    try:
        __import__("repleafgbm_native")
        print("repleafgbm_native already importable", flush=True)
    except ImportError:
        if shutil.which("cargo") is None:
            subprocess.run(
                "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
                "| sh -s -- -y --profile minimal", shell=True, check=False)
        build_env = {**ENV, "PATH": os.path.expanduser("~/.cargo/bin")
                     + os.pathsep + os.environ.get("PATH", "")}
        rc = _run([sys.executable, "-m", "pip", "install", "-q", "./native"],
                  cwd=REPO, env=build_env).returncode
        print("native build:", "ok" if rc == 0 else "FAILED (NumPy fallback)",
              flush=True)
    # Sequential dataset prefetch: fills the sklearn/OpenML cache so parallel
    # workers never fetch (or race) the network.
    sys.path.insert(0, f"{REPO}/src")
    sys.path.insert(0, REPO)
    from benchmarks.suites import get_suite, load
    for suite in spec["suites"]:
        for ds in get_suite(suite).datasets:
            t0 = time.time()
            try:
                load(ds)
                print(f"prefetch {suite}/{ds.name}: {time.time()-t0:.1f}s",
                      flush=True)
            except Exception as exc:  # noqa: BLE001 - report, workers will retry
                print(f"prefetch {suite}/{ds.name} FAILED: {exc}", flush=True)
    os.makedirs(LEDGER_DIR, exist_ok=True)
    with open(SETUP_MARKER, "w") as fh:
        fh.write("ok")
    print("setup complete", flush=True)


def run_task(i, task, n_trials, max_rows):
    worker_ledger = f"{LEDGER_DIR}/worker_{i}.jsonl"
    if os.path.exists(worker_ledger):
        os.remove(worker_ledger)
    cmd = [sys.executable, "-m", "benchmarks.leaderboard",
           "--suite", task["suite"], "--datasets", task["dataset"],
           "--seed-list", str(task["seed"]), "--n-trials", str(n_trials),
           "--max-rows", str(max_rows),
           "--ledger", worker_ledger, "--out", f"/content/out_{i}.md"]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REPO, env=ENV,
                          capture_output=True, text=True)
    ok = proc.returncode == 0
    print(f"[{'done' if ok else 'FAIL'}] {task['suite']}/{task['dataset']} "
          f"seed={task['seed']} ({time.time()-t0:.0f}s)", flush=True)
    if not ok:
        print(proc.stdout[-1500:], flush=True)
        print(proc.stderr[-1500:], file=sys.stderr, flush=True)
    return i, task, ok


def ensure_family_deps():
    """All five model families must be importable, or the leaderboard silently
    skips a family and tasks never complete (the TPU-runtime image, unlike the
    GPU/CPU ones, ships without lightgbm — caught in production on wave 0)."""
    missing = []
    for mod, pkg in [("lightgbm", "lightgbm>=4"), ("xgboost", "xgboost>=2.0"),
                     ("catboost", "catboost>=1.2"), ("optuna", "optuna>=3")]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        _run([sys.executable, "-m", "pip", "install", "-q", *missing],
             check=False)
        print(f"installed missing family deps: {missing}", flush=True)


def wave(spec):
    ensure_family_deps()
    tasks = spec["tasks"]
    results = []
    # Self-adapt to the host we actually got (runtime fallback may land on a
    # smaller machine than the requested worker count assumes).
    n_workers = max(1, min(spec["workers"], os.cpu_count() or 1))
    if n_workers != spec["workers"]:
        print(f"host has {os.cpu_count()} cpus -> {n_workers} workers",
              flush=True)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = [pool.submit(run_task, i, t, spec["n_trials"],
                            spec.get("max_rows", 20000))
                for i, t in enumerate(tasks)]
        for fut in as_completed(futs):
            results.append(fut.result())
    # Merge each worker's private ledger into its suite master (data lines
    # only; the Ledger loader ignores lines without a "key", so _meta headers
    # from the first-ever merge are harmless).
    merged = 0
    for i, task, ok in results:
        worker_ledger = f"{LEDGER_DIR}/worker_{i}.jsonl"
        if not os.path.exists(worker_ledger):
            continue
        master = f"{LEDGER_DIR}/{task['suite']}.jsonl"
        with open(worker_ledger) as src, open(master, "a") as dst:
            for line in src:
                if line.strip():
                    dst.write(line)
                    merged += 1
        os.remove(worker_ledger)
    n_ok = sum(1 for *_, ok in results if ok)
    print(f"wave complete: {n_ok}/{len(tasks)} tasks ok, "
          f"{merged} ledger lines merged", flush=True)


def main():
    with open(SPEC) as fh:
        spec = json.load(fh)
    if spec["mode"] == "setup" or not os.path.exists(SETUP_MARKER):
        setup(spec)
    if spec["mode"] == "wave":
        wave(spec)


main()
