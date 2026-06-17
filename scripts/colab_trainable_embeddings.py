"""On-VM driver for the trainable-embeddings benchmark (run via ``colab exec -f``).

Mirrors ``scripts/colab_remote_test.py``: the Colab CLI reads this file locally
and executes it in the remote kernel. It expects the working tree to have been
uploaded to ``/content/rlgbm.tar.gz`` (see
``scripts/colab_trainable_embeddings.sh``). It then:

  1. extracts the repo to ``/content/repleafgbm``,
  2. ensures torch is importable (Colab runtimes ship it) and best-effort
     installs the optional external GBMs,
  3. runs ``benchmarks/trainable_embeddings.py`` at full settings,
  4. tars the artifacts to ``/content/te_results.tar.gz`` for download.

Pretraining of the ``torch_*`` encoders runs on CPU (fit-time only); the VM is
used for scale, isolation, and a torch-equipped reproducible environment — not
for GPU-accelerated pretraining.
"""

import os
import subprocess
import sys
import tarfile

REPO = "/content/repleafgbm"
TARBALL = "/content/rlgbm.tar.gz"
OUT = "/content/te_artifacts"
RESULTS_TARBALL = "/content/te_results.tar.gz"
ENV = {**os.environ, "OMP_NUM_THREADS": "1", "PYTHONPATH": f"{REPO}/src"}


def _run(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, **kw)


def extract_repo():
    os.makedirs(REPO, exist_ok=True)
    with tarfile.open(TARBALL, "r:gz") as tf:
        tf.extractall(REPO)
    print(f"extracted working tree to {REPO}", flush=True)


def ensure_torch():
    try:
        import torch
    except ImportError:
        _run([sys.executable, "-m", "pip", "install", "-q", "torch"], check=True)
        import torch
    print(f"torch {torch.__version__} (cuda_available={torch.cuda.is_available()}; "
          "encoder pretraining runs on CPU)", flush=True)


def ensure_external():
    """Best-effort: external GBMs are optional benchmark comparisons."""
    for pkg in ("lightgbm", "xgboost", "catboost"):
        try:
            __import__(pkg)
        except ImportError:
            _run([sys.executable, "-m", "pip", "install", "-q", pkg])


def run_benchmark():
    proc = _run(
        [sys.executable, "benchmarks/trainable_embeddings.py",
         "--seeds", "5", "--n-estimators", "200", "--epochs", "30",
         "--out-dir", OUT],
        cwd=REPO, env=ENV,
    )
    return proc.returncode


def main():
    extract_repo()
    ensure_torch()
    ensure_external()
    rc = run_benchmark()
    if os.path.isdir(OUT):
        with tarfile.open(RESULTS_TARBALL, "w:gz") as tf:
            tf.add(OUT, arcname="trainable_embeddings")
        print(f"wrote {RESULTS_TARBALL}", flush=True)
    else:
        print(f"benchmark produced no artifacts at {OUT}", file=sys.stderr, flush=True)
    if rc != 0:
        sys.exit(rc)


main()
