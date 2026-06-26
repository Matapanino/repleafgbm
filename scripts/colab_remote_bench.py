"""On-VM driver for the fair-leaderboard CPU run (run via ``colab exec -f``).

The Colab CLI reads this file locally and executes its contents in the remote
CPU kernel. It expects (see ``scripts/colab_cpu_bench.sh``):

  * the working tree at ``/content/rlgbm.tar.gz``,
  * leaderboard argv (one per line) at ``/content/bench_argv.txt``,
  * optionally a prior ledger at ``/content/ledger_in.jsonl`` (to resume).

It then extracts the repo, installs the ``[bench]`` stack (Optuna / SciPy /
Matplotlib / the external GBMs) best-effort, restores the ledger so completed
cells are skipped, runs ``benchmarks.leaderboard`` single-threaded, and leaves
the report + ledger (+ CD PNGs) under ``/content`` for the launcher to download.
``OMP_NUM_THREADS=1`` avoids the torch+libomp deadlock and steadies timing.
"""

import os
import shutil
import subprocess
import sys
import tarfile

REPO = "/content/repleafgbm"
TARBALL = "/content/rlgbm.tar.gz"
ARGV_FILE = "/content/bench_argv.txt"
LEDGER_IN = "/content/ledger_in.jsonl"
LEDGER = "/content/ledger.jsonl"
ENV = {**os.environ, "OMP_NUM_THREADS": "1", "PYTHONPATH": f"{REPO}/src"}


def _run(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, **kw)


def extract_repo():
    os.makedirs(REPO, exist_ok=True)
    with tarfile.open(TARBALL, "r:gz") as tf:
        try:
            tf.extractall(REPO, filter="data")  # py>=3.12; silences the 3.14 warning
        except TypeError:  # older VM Python without the filter kwarg
            tf.extractall(REPO)
    print(f"extracted working tree to {REPO}", flush=True)


def ensure_deps():
    # Colab ships numpy/pandas/scikit-learn/scipy/lightgbm/xgboost/matplotlib;
    # optuna + catboost usually need installing. Best-effort, quiet.
    _run([sys.executable, "-m", "pip", "install", "-q",
          "optuna>=3", "catboost>=1.2", "xgboost>=2.0"], check=False)


def ensure_rust_native():
    """Build the Rust split backend so RepLeaf's split_backend='auto' uses it
    (RepLeaf dominates the HPO cost on the NumPy backend). Rust⇄NumPy results are
    bitwise-identical, so this only changes speed; on any failure RepLeaf falls
    back to NumPy automatically."""
    try:
        import repleafgbm_native  # noqa: F401
        print("repleafgbm_native already importable", flush=True)
        return
    except ImportError:
        pass
    if shutil.which("cargo") is None:
        print("+ installing rust toolchain (rustup, minimal)", flush=True)
        subprocess.run(
            "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
            "| sh -s -- -y --profile minimal", shell=True, check=False)
    build_env = {**ENV, "PATH": os.path.expanduser("~/.cargo/bin") + os.pathsep
                 + os.environ.get("PATH", "")}
    print("+ pip install ./native  (maturin --release; ~5-10 min)", flush=True)
    rc = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "./native"],
                        cwd=REPO, env=build_env).returncode
    ok = rc == 0 and subprocess.run(
        [sys.executable, "-c", "import repleafgbm_native"],
        cwd=REPO, env=build_env).returncode == 0
    print("repleafgbm_native built -> RepLeaf uses Rust backend" if ok
          else "WARN: native build failed -> RepLeaf uses NumPy fallback",
          flush=True)


def read_argv():
    with open(ARGV_FILE, encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f if ln.strip()]


def main():
    extract_repo()
    ensure_deps()
    ensure_rust_native()
    if os.path.exists(LEDGER_IN):
        shutil.copy(LEDGER_IN, LEDGER)
        # Consume it: on a kept-alive VM the next slice must NOT re-copy this
        # (now-stale) upload over the ledger that accumulates across slices.
        os.remove(LEDGER_IN)
        print(f"resuming from {LEDGER_IN} (restored once)", flush=True)

    argv = read_argv()
    print(f"leaderboard argv: {argv}", flush=True)
    # --out is /content/leaderboard.md, so the CD PNGs land in /content already
    # (assets_dir = report dir); the launcher downloads them directly.
    proc = _run([sys.executable, "-m", "benchmarks.leaderboard", *argv],
                cwd=REPO, env=ENV)
    # Greppable marker for the launcher log; avoids the IPython "SystemExit"
    # noise a bare sys.exit(0) produces under `colab exec`.
    if proc.returncode != 0:
        print(f"REMOTE_FAIL rc={proc.returncode}", flush=True)
        sys.exit(proc.returncode)
    print("REMOTE_OK", flush=True)


if __name__ == "__main__":
    main()
