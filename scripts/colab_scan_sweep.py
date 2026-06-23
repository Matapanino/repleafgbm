"""On-VM driver for the CUDA adaptive-scan-threshold sweep (run via ``colab exec -f``).

Companion to ``scripts/colab_remote_test.py`` (parity loop). It bootstraps the same
way — extract the uploaded working tree, ensure CuPy — then sweeps the private
``REPLEAFGBM_CUDA_SCAN_MIN_CELLS`` override across the workloads the split scan
dominates, recording one ``benchmarks.gpu_profile`` JSONL row per (workload,
threshold) to ``/content/gpu_bench/scan_sweep.jsonl`` and a per-workload crossover
table to ``/content/scan_sweep_report.md``.

Only the cuda backend reads the threshold, so this sweep is **cuda-only**. It does
not change any default — it produces the measurement a default change would need
(the verdict is a separate, results-analyst step). A discarded cuda fit warms the
CuPy kernel JIT before the timed rows so the first threshold is not penalised.
"""

import json
import os
import subprocess
import sys
import tarfile
from collections import defaultdict

REPO = "/content/repleafgbm"
TARBALL = "/content/rlgbm.tar.gz"
SWEEP_JSONL = "/content/gpu_bench/scan_sweep.jsonl"
REPORT = "/content/scan_sweep_report.md"
ENV = {**os.environ, "OMP_NUM_THREADS": "1", "PYTHONPATH": f"{REPO}/src"}

# (task, n_train, n_features, leaf_model, extra_args). Mirrors the parity loop's
# narrow (30f, host-scan regime) / wide (200f, GPU-scan regime) shapes plus the
# scan-dominated multiclass-c5 case (split scan is ~85% of its fit).
WORKLOADS = [
    ("regression", 50_000, 30, "constant", []),
    ("regression", 30_000, 200, "embedded_linear", []),
    ("binary", 50_000, 30, "constant", []),
    ("binary", 30_000, 200, "embedded_linear", []),
    ("multiclass", 30_000, 200, "constant", ["--n-classes", "5"]),
]
# 0 = every node on the GPU scan; very_large = every node on the host scan;
# 32768 is the current default (_GPU_SCAN_MIN_CELLS).
THRESHOLDS = ["0", "8192", "32768", "131072", "very_large"]


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


def warmup():
    """One throwaway cuda fit so the CuPy kernel JIT is paid before timed rows."""
    sys.path.insert(0, f"{REPO}/src")
    import numpy as np

    from repleafgbm import RepLeafRegressor

    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 30))
    y = X[:, 0] + rng.normal(0.0, 0.1, 2000)
    RepLeafRegressor(
        n_estimators=5, num_leaves=8, leaf_model="constant",
        split_backend="cuda", random_state=0,
    ).fit(X, y)
    print("cuda JIT warmup done", flush=True)


def run_sweep():
    if os.path.exists(SWEEP_JSONL):
        os.remove(SWEEP_JSONL)  # fresh window
    for task, n_train, n_features, leaf, extra in WORKLOADS:
        _run(
            [sys.executable, "-m", "benchmarks.gpu_profile",
             "--task", task, "--backend", "cuda", "--leaf-model", leaf,
             "--n-train", str(n_train), "--n-test", "10000",
             "--n-features", str(n_features), "--n-estimators", "30",
             "--scan-min-cells-sweep", *THRESHOLDS,
             "--out", SWEEP_JSONL, *extra],
            cwd=REPO, env=ENV, check=True,
        )
    with open(SWEEP_JSONL) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _thr_label(thr):
    return "very_large" if thr >= 1_000_000_000 else str(thr)


def write_report(rows, gpu):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["task"], r["n_features"], r.get("n_classes"))].append(r)

    lines = [
        "# CUDA adaptive-scan threshold sweep",
        "",
        f"- GPU: **{gpu}**",
        "- Default threshold: **32768** (`_GPU_SCAN_MIN_CELLS`); override is the "
        "private `REPLEAFGBM_CUDA_SCAN_MIN_CELLS`.",
        "",
        "Per workload, `benchmarks.gpu_profile` cuda fit (30 trees) across "
        "`REPLEAFGBM_CUDA_SCAN_MIN_CELLS` ∈ {0, 8192, 32768, 131072, very_large}. "
        "`0` forces every node onto the GPU scan; `very_large` forces the host "
        "scan. Lower fit is better; **bold** = fastest threshold for the workload. "
        "Measurement only — not a default change.",
        "",
    ]
    headline = []
    for (task, nf, nc), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        rs.sort(key=lambda r: r["cuda_scan_min_cells"])
        best = min(rs, key=lambda r: r["fit_seconds"])
        default = next((r for r in rs if r["cuda_scan_min_cells"] == 32768), None)
        title = task + (f" K={nc}" if task == "multiclass" else "") + f", {nf}f"
        if default is not None and default["fit_seconds"]:
            delta = (default["fit_seconds"] - best["fit_seconds"]) / default["fit_seconds"]
            headline.append(
                f"- **{title}**: best={_thr_label(best['cuda_scan_min_cells'])} "
                f"({best['fit_seconds']:.2f}s) vs default 32768 "
                f"({default['fit_seconds']:.2f}s) → {delta:+.1%} headroom"
            )
        lines += [
            f"## {title}",
            "",
            "| threshold | fit (s) | split_scan (s) | n_small | n_gpu | quality |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for r in rs:
            tb = r.get("transfer_bytes") or {}
            ps = r.get("phase_seconds") or {}
            ss = ps.get("split_scan")
            ss_s = f"{ss:.2f}" if isinstance(ss, (int, float)) else "-"
            q = ", ".join(f"{k}={v:.4g}" for k, v in (r.get("quality") or {}).items())
            fit_s = f"{r['fit_seconds']:.2f}"
            if r is best:
                fit_s = f"**{fit_s}**"
            lines.append(
                f"| {_thr_label(r['cuda_scan_min_cells'])} | {fit_s} | {ss_s} | "
                f"{tb.get('n_small_scans', 0)} | {tb.get('n_gpu_scans', 0)} | {q} |"
            )
        lines.append("")
    lines = lines[:7] + ["## Headline (fastest threshold vs default 32768)", "",
                         *headline, ""] + lines[7:]
    with open(REPORT, "w") as fh:
        fh.write("\n".join(lines))
    return lines


def main():
    extract_repo()
    gpu = ensure_cupy()
    warmup()
    rows = run_sweep()
    lines = write_report(rows, gpu)
    print("\n".join(lines), flush=True)
    print(f"\nwrote {REPORT}", flush=True)


main()
