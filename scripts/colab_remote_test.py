"""On-VM driver for the CUDA backend dev loop (run via ``colab exec -f``).

The Colab CLI reads this file locally and executes its contents in the remote
GPU kernel. It expects the working tree to have been uploaded to
``/content/rlgbm.tar.gz`` (see ``scripts/colab_gpu_test.sh``). It then:

  1. extracts the repo to ``/content/repleafgbm``,
  2. ensures CuPy is importable (Colab GPU runtimes ship it),
  3. runs the CUDA parity subset (``tests/test_cuda_backend.py``) single-threaded,
  4. micro-benchmarks GPU vs NumPy histogram build,
  5. writes a markdown report to ``/content/cuda_parity_report.md``.

It deliberately runs only the CUDA test module so torch / lightgbm are never
imported (avoids the libomp deadlock) and the loop stays fast.
"""

import os
import subprocess
import sys
import tarfile
import time

REPO = "/content/repleafgbm"
TARBALL = "/content/rlgbm.tar.gz"
REPORT = "/content/cuda_parity_report.md"
ENV = {**os.environ, "OMP_NUM_THREADS": "1", "PYTHONPATH": f"{REPO}/src"}


def _run(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, **kw)


def extract_repo():
    os.makedirs(REPO, exist_ok=True)
    with tarfile.open(TARBALL, "r:gz") as tf:
        tf.extractall(REPO)
    print(f"extracted working tree to {REPO}", flush=True)


def ensure_cupy():
    try:
        import cupy  # noqa: F401
    except ImportError:
        _run([sys.executable, "-m", "pip", "install", "-q", "cupy-cuda12x"], check=True)
    import cupy
    dev = cupy.cuda.runtime.getDeviceProperties(0)
    name = dev["name"].decode() if isinstance(dev["name"], bytes) else dev["name"]
    print(f"CuPy {cupy.__version__} on {name}", flush=True)
    return name


def run_parity_tests():
    proc = _run(
        [sys.executable, "-m", "pytest", "tests/test_cuda_backend.py", "-q"],
        cwd=REPO, env=ENV, capture_output=True, text=True,
    )
    print(proc.stdout, flush=True)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, flush=True)
    return proc.returncode, proc.stdout.strip().splitlines()[-1] if proc.stdout else ""


def micro_benchmark():
    sys.path.insert(0, f"{REPO}/src")
    import numpy as np

    from repleafgbm.backends import CudaSplitBackend, NumPySplitBackend

    n, F, B = 200_000, 50, 65
    rng = np.random.default_rng(0)
    binned = rng.integers(0, B - 1, size=(n, F)).astype(np.uint16)
    rows = np.arange(n, dtype=np.int64)
    grad = rng.normal(size=n)
    hess = np.abs(rng.normal(size=n)) + 0.1
    np_b, cu_b = NumPySplitBackend(), CudaSplitBackend()

    cu_b.build_histograms(binned, rows[:1000], grad, hess, B)  # warm up JIT

    def timeit(backend, reps=5):
        t = time.perf_counter()
        for _ in range(reps):
            backend.build_histograms(binned, rows, grad, hess, B)
        return (time.perf_counter() - t) / reps

    t_np = timeit(np_b)
    t_cu = timeit(cu_b)
    return n, F, B, t_np, t_cu


def end_to_end_benchmark():
    """Full RepLeafRegressor.fit, numpy vs cuda backend.

    This is the number that actually matters for sizing further GPU work:
    histogram build is on the GPU (cuda backend) but tree growth, split scan,
    and leaf fitting (leaf_linear_stats) stay on the host. A large end-to-end
    speedup means histogram dominated; a modest one means host work (incl. leaf
    fitting) is now the bottleneck — i.e. whether Phase C1 is worth it.
    """
    sys.path.insert(0, f"{REPO}/src")
    import numpy as np

    from repleafgbm import RepLeafRegressor

    n, d = 100_000, 30
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, d))
    y = (
        np.where(X[:, 0] > 0.0, 2.0, -2.0)
        + X[:, 1] - X[:, 2] + rng.normal(0.0, 0.1, n)
    )

    def fit_time(backend):
        t = time.perf_counter()
        RepLeafRegressor(
            n_estimators=50, num_leaves=31, leaf_model="embedded_linear",
            split_backend=backend, random_state=0,
        ).fit(X, y)
        return time.perf_counter() - t

    fit_time("cuda")  # warm up JIT + caches
    t_np = fit_time("numpy")
    t_cu = fit_time("cuda")
    return n, d, t_np, t_cu


def main():
    extract_repo()
    gpu = ensure_cupy()
    rc, summary = run_parity_tests()
    n, F, B, t_np, t_cu = micro_benchmark()
    e2e_n, e2e_d, e2e_np, e2e_cu = end_to_end_benchmark()
    speedup = (t_np / t_cu) if t_cu else float("nan")
    e2e_speedup = (e2e_np / e2e_cu) if e2e_cu else float("nan")

    lines = [
        "# CUDA backend parity report",
        "",
        f"- GPU: **{gpu}**",
        f"- Parity tests (`tests/test_cuda_backend.py`): "
        f"**{'PASS' if rc == 0 else 'FAIL'}** — `{summary}`",
        "",
        "## Histogram micro-benchmark",
        "",
        f"Single `build_histograms` over {n:,} rows x {F} features x {B} bins "
        "(mean of 5):",
        "",
        f"- NumPy: **{t_np * 1e3:.2f} ms**",
        f"- CUDA:  **{t_cu * 1e3:.2f} ms**",
        f"- Speedup: **{speedup:.2f}x**",
        "",
        "_Phase B1: binned is uploaded once and cached on-device (keyed by "
        "identity); each call ships only its rows + gathered grad/hess, so "
        "repeated builds over the same matrix avoid re-transferring it._",
        "",
        "## End-to-end training (the number that sizes Phase C)",
        "",
        f"`RepLeafRegressor.fit` over {e2e_n:,} rows x {e2e_d} features, 50 "
        "trees, embedded_linear (GPU histogram + host leaf fitting):",
        "",
        f"- numpy backend: **{e2e_np:.2f} s**",
        f"- cuda backend:  **{e2e_cu:.2f} s**",
        f"- Speedup: **{e2e_speedup:.2f}x**",
        "",
        "_A large speedup ⇒ histogram dominated, so Phase C1 (GPU leaf stats) "
        "adds little; a modest one ⇒ host leaf fitting is now the bottleneck "
        "and C1 is justified._",
        "",
    ]
    with open(REPORT, "w") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines), flush=True)
    print(f"\nwrote {REPORT}", flush=True)
    if rc != 0:
        sys.exit(rc)


main()
