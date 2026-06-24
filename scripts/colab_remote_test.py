"""On-VM driver for the CUDA backend dev loop (run via ``colab exec -f``).

The Colab CLI reads this file locally and executes its contents in the remote
GPU kernel. It expects the working tree to have been uploaded to
``/content/rlgbm.tar.gz`` (see ``scripts/colab_gpu_test.sh``). It then:

  1. extracts the repo to ``/content/repleafgbm``,
  2. ensures CuPy is importable (Colab GPU runtimes ship it),
  3. runs the CUDA parity subset (``tests/test_cuda_backend.py``) single-threaded,
  4. micro-benchmarks GPU vs NumPy histogram build,
  5. runs the ``benchmarks.gpu_profile`` matrix (numpy vs cuda across
     regression/binary/multiclass/multioutput at narrow/wide shapes), recording
     per-fit transfer volume to ``/content/gpu_bench/cases.jsonl``, plus a
     multi-output on-device-scan A/B (device path off vs on),
  6. writes a parity report to ``/content/cuda_parity_report.md`` and a
     standalone backend-comparison suite to ``/content/gpu_backend_suite.md``.

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
BACKEND_SUITE = "/content/gpu_backend_suite.md"
GPU_BENCH = "/content/gpu_bench/cases.jsonl"
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


def end_to_end_benchmark(n, d):
    """Full RepLeafRegressor.fit, numpy vs cuda backend, for n rows x d features.

    histogram build is on the GPU (cuda backend) but tree growth, the categorical
    scan, and leaf fitting (leaf_linear_stats) stay on the host. Phase B2 keeps
    the histogram resident and runs the *numeric* split scan on-device, so its
    value grows with the per-node histogram size — wide d / many bins is where it
    should beat the B1 round-trip (and narrow d its worst case).
    """
    sys.path.insert(0, f"{REPO}/src")
    import numpy as np

    from repleafgbm import RepLeafRegressor

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
    return t_np, t_cu


def gpu_profile_smoke():
    """Run the ``benchmarks.gpu_profile`` matrix (numpy vs cuda across tasks and
    narrow/wide shapes) and return the parsed JSONL rows. Each task contributes a
    numpy/cuda pair at a narrow (30f, B2's worst case) and a wide (200f, B2's
    sweet spot) shape, so the backend-comparison summary can show the cuda
    speedup where it matters. The cuda rows also carry the per-fit transfer
    counters that motivate the next optimization (device-resident grad/hess)."""
    import json

    # (task, n_train, n_features, leaf_model, extra_args)
    shapes = [
        ("regression", 50_000, 30, "constant", []),
        ("regression", 30_000, 200, "embedded_linear", []),
        ("binary", 50_000, 30, "constant", []),
        ("binary", 30_000, 200, "embedded_linear", []),
        ("multiclass", 30_000, 200, "constant", ["--n-classes", "5"]),
        ("multioutput", 50_000, 30, "constant", ["--n-outputs", "5"]),
        ("multioutput", 30_000, 200, "embedded_linear", ["--n-outputs", "5"]),
    ]
    for task, n_train, n_features, leaf, extra in shapes:
        for backend in ("numpy", "cuda"):
            _run(
                [sys.executable, "-m", "benchmarks.gpu_profile",
                 "--task", task, "--backend", backend, "--leaf-model", leaf,
                 "--n-train", str(n_train), "--n-test", "10000",
                 "--n-features", str(n_features), "--n-estimators", "30",
                 "--out", GPU_BENCH, *extra],
                cwd=REPO, env=ENV, check=True,
            )
    with open(GPU_BENCH) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def multioutput_device_ab():
    """Paired A/B for the multi-output on-device scan: cuda with the device path
    off (host stack + host scan — the pre-device baseline) vs on, toggled via
    ``REPLEAFGBM_CUDA_MO_DEVICE_SCAN`` and interleaved per shape to limit drift.

    Returns ``(case, n_features, fit_off, fit_on, histscan_off, histscan_on,
    hist_d2h_off, winner_d2h_on)`` per shape. ``histscan`` sums the ``histogram``
    and ``split_scan`` phase seconds — the two phases the device path moves onto
    the GPU; the on-device path should shrink them and replace the per-output
    histogram D2H copies with a 32-byte winner pack."""
    import json

    shapes = [
        ("multioutput", 50_000, 30, "constant", ["--n-outputs", "5"]),
        ("multioutput", 30_000, 200, "embedded_linear", ["--n-outputs", "5"]),
    ]
    results = []
    for task, n_train, n_features, leaf, extra in shapes:
        per_gate = {}
        for gate in ("0", "1"):  # off (baseline) then on (device path)
            out = f"/content/gpu_bench/mo_ab_{n_features}_{gate}.jsonl"
            if os.path.exists(out):
                os.remove(out)
            _run(
                [sys.executable, "-m", "benchmarks.gpu_profile",
                 "--task", task, "--backend", "cuda", "--leaf-model", leaf,
                 "--n-train", str(n_train), "--n-test", "10000",
                 "--n-features", str(n_features), "--n-estimators", "30",
                 "--out", out, *extra],
                cwd=REPO,
                env={**ENV, "REPLEAFGBM_CUDA_MO_DEVICE_SCAN": gate},
                check=True,
            )
            with open(out) as fh:
                per_gate[gate] = json.loads(fh.readlines()[-1])
        off, on = per_gate["0"], per_gate["1"]
        ps_off, ps_on = off.get("phase_seconds", {}), on.get("phase_seconds", {})
        tb_off, tb_on = off.get("transfer_bytes", {}), on.get("transfer_bytes", {})
        results.append((
            off["case_id"].rsplit("_", 1)[0], n_features,
            off["fit_seconds"], on["fit_seconds"],
            ps_off.get("histogram", 0.0) + ps_off.get("split_scan", 0.0),
            ps_on.get("histogram", 0.0) + ps_on.get("split_scan", 0.0),
            tb_off.get("hist_d2h_bytes", 0), tb_on.get("winner_d2h_bytes", 0),
        ))
    return results


def backend_comparison(bench_rows):
    """Pair numpy/cuda rows by case (case_id minus the backend suffix) and return
    ``(key, n_features, task, np_fit, cu_fit, np_pred, cu_pred)`` tuples sorted by
    feature width — the heart of the backend-suite report."""
    by_key = {}
    for r in bench_rows:
        key = r["case_id"].rsplit("_", 1)[0]  # strip "_<backend>"
        by_key.setdefault(key, {})[r["backend"]] = r
    out = []
    for key, pair in by_key.items():
        if "numpy" not in pair or "cuda" not in pair:
            continue
        npr, cur = pair["numpy"], pair["cuda"]
        out.append((key, npr["n_features"], npr["task"],
                    npr["fit_seconds"], cur["fit_seconds"],
                    npr["predict_seconds"], cur["predict_seconds"]))
    return sorted(out, key=lambda t: (t[2], t[1]))


def write_backend_suite(bench_rows, gpu):
    """Standalone backend-comparison report (numpy vs cuda fit/predict speedups
    across the task x shape matrix), written to ``BACKEND_SUITE``."""
    rows = backend_comparison(bench_rows)
    lines = [
        "# GPU backend suite — numpy vs cuda",
        "",
        f"- GPU: **{gpu}**",
        "",
        "`benchmarks.gpu_profile`, 30 trees, each task at a narrow (30f) and wide "
        "(200f) shape. Speedup = numpy / cuda (higher is better for cuda). The "
        "cuda histogram + on-device numeric scan (Phase B2) pays off on the wide "
        "shapes where the per-node histogram is large.",
        "",
        "| case | task | features | numpy fit (s) | cuda fit (s) | fit speedup | "
        "numpy pred (s) | cuda pred (s) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for key, nf, task, np_f, cu_f, np_p, cu_p in rows:
        sp = (np_f / cu_f) if cu_f else float("nan")
        lines.append(
            f"| {key} | {task} | {nf} | {np_f:.2f} | {cu_f:.2f} | **{sp:.2f}x** | "
            f"{np_p:.3f} | {cu_p:.3f} |"
        )
    lines.append("")
    with open(BACKEND_SUITE, "w") as fh:
        fh.write("\n".join(lines))
    return lines


def main():
    extract_repo()
    gpu = ensure_cupy()
    rc, summary = run_parity_tests()
    n, F, B, t_np, t_cu = micro_benchmark()
    speedup = (t_np / t_cu) if t_cu else float("nan")
    bench_rows = gpu_profile_smoke()

    # Narrow (B2's worst case) and wide (B2's intended sweet spot — the per-node
    # histogram copy B2 keeps resident is biggest here) end-to-end fits.
    configs = [("narrow", 100_000, 30), ("wide", 50_000, 200)]
    e2e = []
    for label, e2e_n, e2e_d in configs:
        np_s, cu_s = end_to_end_benchmark(e2e_n, e2e_d)
        e2e.append((label, e2e_n, e2e_d, np_s, cu_s, (np_s / cu_s) if cu_s else float("nan")))

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
        "_Phase B1/B2: binned is uploaded once and cached on-device, and the "
        "histogram is returned resident (no per-build copy back)._",
        "",
        "## End-to-end training (Phase B2: resident hist + GPU numeric scan)",
        "",
        "`RepLeafRegressor.fit`, 50 trees, embedded_linear (GPU histogram + GPU "
        "numeric scan; host categorical scan + leaf fitting):",
        "",
        "| config | rows x feat | numpy (s) | cuda (s) | speedup |",
        "| --- | --- | --- | --- | --- |",
    ]
    for label, e2e_n, e2e_d, np_s, cu_s, sp in e2e:
        lines.append(
            f"| {label} | {e2e_n:,} x {e2e_d} | {np_s:.2f} | {cu_s:.2f} | "
            f"**{sp:.2f}x** |"
        )
    lines += [
        "",
        "_B2's value grows with per-node histogram size: narrow d is its worst "
        "case (tiny scan, GPU launch/sync overhead), wide d its best (the big "
        "per-node histogram round-trip B1 paid is now avoided)._",
        "",
        "## Per-fit transfer counters (`benchmarks.gpu_profile`)",
        "",
        "End-to-end `gpu_profile` smoke; transfer columns are the CUDA backend's "
        "private H2D/D2H byte counters for one fit (numpy reports none). The "
        "grad/hess H2D column is the per-node host gather the next optimization "
        "targets — full rows saved to `gpu_bench/cases.jsonl`.",
        "",
        "| case_id | backend | fit (s) | binned H2D | grad/hess H2D | "
        "winner D2H | hist D2H |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in bench_rows:
        tb = r.get("transfer_bytes") or {}
        lines.append(
            f"| {r['case_id']} | {r['backend']} | {r['fit_seconds']:.2f} | "
            f"{tb.get('binned_h2d_bytes', 0):,} | "
            f"{tb.get('gradhess_h2d_bytes', 0):,} | "
            f"{tb.get('winner_d2h_bytes', 0):,} | "
            f"{tb.get('hist_d2h_bytes', 0):,} |"
        )
    lines += [
        "",
        "_Expect `binned_uploads == 1` per fit (Phase B1 cache) and a non-zero "
        "grad/hess H2D total — that gather is what a device-resident grad/hess "
        "buffer would remove (docs/gpu_roadmap.md, Phase 1)._",
        "",
    ]

    # Multi-output on-device scan A/B (the device residency this change adds).
    mo_ab = multioutput_device_ab()
    lines += [
        "## Multi-output device scan A/B (`REPLEAFGBM_CUDA_MO_DEVICE_SCAN`)",
        "",
        "cuda multi-output fit with the on-device summed-gain scan **off** (host "
        "stack + host scan — the pre-device baseline) vs **on**. `hist+scan` sums "
        "the `histogram`+`split_scan` phase seconds (the two phases the device "
        "path keeps on the GPU); on-device should shrink them and replace the "
        "per-output histogram D2H with a 32-byte winner pack.",
        "",
        "| case | features | fit off (s) | fit on (s) | speedup | hist+scan off "
        "(s) | hist+scan on (s) | hist D2H off | winner D2H on |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for case, nf, f_off, f_on, hs_off, hs_on, d2h_off, win_on in mo_ab:
        sp = (f_off / f_on) if f_on else float("nan")
        lines.append(
            f"| {case} | {nf} | {f_off:.2f} | {f_on:.2f} | **{sp:.2f}x** | "
            f"{hs_off:.2f} | {hs_on:.2f} | {d2h_off:,} | {win_on:,} |"
        )
    lines += [
        "",
        "_Parity is covered by `tests/test_cuda_backend.py` (allclose); this table "
        "is the speed verdict — the device path must win (or at least not regress) "
        "on the wide shape, with the narrow shape protected by the adaptive "
        "small-scan crossover._",
        "",
    ]

    with open(REPORT, "w") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines), flush=True)
    print(f"\nwrote {REPORT}", flush=True)

    # Standalone backend-comparison suite (numpy vs cuda across the matrix).
    suite_lines = write_backend_suite(bench_rows, gpu)
    print("\n".join(suite_lines), flush=True)
    print(f"\nwrote {BACKEND_SUITE}", flush=True)
    if rc != 0:
        sys.exit(rc)


main()
