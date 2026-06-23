"""On-VM driver for the CUDA scan-threshold A/B confirmation (run via ``colab exec -f``).

Follow-up to ``scripts/colab_scan_sweep.py``. The sweep (sequential, single-sample)
hinted that the *host* numeric scan is ~5-11% faster than the *on-device* scan on
wide 200f shapes, but that delta sat near the run-to-run noise. This driver
confirms it with an **interleaved, paired A/B**: per rep it times threshold A
(32768 → GPU scan at 200f) and threshold B (131072 → host scan at 200f)
back-to-back on identical data, alternating order each rep so thermal/cache drift
cancels in the paired difference. It reuses ``benchmarks.gpu_profile.run_case`` (so
phase timings + scan-path counters come from the same harness) in one warm
process, and reports the paired diff (A-B) per workload.

cuda-only; measurement only — it does not change any default. The verdict (keep
32768 vs raise it) is a separate results-analyst step.
"""

import json
import os
import sys
import tarfile
from collections import defaultdict
from statistics import mean, pstdev

REPO = "/content/repleafgbm"
TARBALL = "/content/rlgbm.tar.gz"
AB_JSONL = "/content/gpu_bench/scan_ab.jsonl"
REPORT = "/content/scan_ab_report.md"
os.environ.setdefault("OMP_NUM_THREADS", "1")  # single-thread OMP for stable timing

N_REPS = 5
A, B = 32768, 131072  # A = on-device scan at 200f (51400 cells); B = host scan
# (task, leaf_model, extra argv) — the wide (200f) shapes where the sweep saw the
# host edge; matches the sweep's leaf-model choices for comparability.
WIDE = [
    ("regression", "embedded_linear", []),
    ("binary", "embedded_linear", []),
    ("multiclass", "constant", ["--n-classes", "5"]),
]


def extract_repo():
    os.makedirs(REPO, exist_ok=True)
    with tarfile.open(TARBALL, "r:gz") as tf:
        tf.extractall(REPO)
    print(f"extracted working tree to {REPO}", flush=True)


def ensure_cupy():
    import subprocess
    try:
        import cupy  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "cupy-cuda12x"],
                       check=True)
    import cupy
    dev = cupy.cuda.runtime.getDeviceProperties(0)
    name = dev["name"].decode() if isinstance(dev["name"], bytes) else dev["name"]
    print(f"CuPy {cupy.__version__} on {name}", flush=True)
    return name


def _args_for(gpu_profile, task, leaf, extra, thresh):
    argv = [
        "--task", task, "--backend", "cuda", "--leaf-model", leaf,
        "--n-train", "30000", "--n-test", "10000", "--n-features", "200",
        "--n-estimators", "30", "--cuda-scan-min-cells", str(thresh),
        "--out", AB_JSONL, *extra,
    ]
    return gpu_profile.build_parser().parse_args(argv)


def run():
    # benchmarks/ lives at the repo root, repleafgbm under src/; add both so the
    # in-process `from benchmarks import gpu_profile` and its repleafgbm imports
    # resolve (REPO/src alone found repleafgbm but not the benchmarks package).
    sys.path.insert(0, f"{REPO}/src")
    sys.path.insert(0, REPO)
    from benchmarks import gpu_profile

    # Warm the CuPy kernel JIT + caches with one discarded cuda fit.
    gpu_profile.run_case(_args_for(gpu_profile, "regression", "constant", [], A))
    print("cuda JIT warmup done", flush=True)

    if os.path.exists(AB_JSONL):
        os.remove(AB_JSONL)
    rows = []
    for rep in range(N_REPS):
        # Alternate which threshold goes first each rep to cancel order bias.
        order = [(A, B), (B, A)][rep % 2]
        for task, leaf, extra in WIDE:
            for thresh in order:
                row = gpu_profile.run_case(_args_for(gpu_profile, task, leaf, extra, thresh))
                row["rep"] = rep
                rows.append(row)
                ps = (row.get("phase_seconds") or {}).get("split_scan")
                print(f"rep{rep} {task:11s} thr{thresh:<7d} "
                      f"fit={row['fit_seconds']:.2f}s scan={ps:.2f}s", flush=True)
    os.makedirs(os.path.dirname(AB_JSONL), exist_ok=True)
    with open(AB_JSONL, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return rows


def write_report(rows, gpu):
    # Group by (task, n_classes) → {thresh: [fit per rep]} and paired diffs.
    by_wl = defaultdict(lambda: defaultdict(dict))  # wl -> rep -> {thresh: row}
    for r in rows:
        wl = (r["task"], r.get("n_classes"))
        by_wl[wl][r["rep"]][r["cuda_scan_min_cells"]] = r

    lines = [
        "# CUDA scan-threshold A/B confirmation",
        "",
        f"- GPU: **{gpu}**",
        f"- A = **{A}** (on-device scan at 200f) vs B = **{B}** (host scan); "
        f"{N_REPS} interleaved reps, identical data, paired diff `A - B` "
        "(positive ⇒ host faster). 200f, 30 trees, cuda backend.",
        "- Measurement only — not a default change.",
        "",
        "| workload | A=32768 mean (s) | B=131072 mean (s) | paired Δ(A-B) mean | "
        "paired Δ % | host faster | signal |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for (task, nc), reps in sorted(by_wl.items(), key=lambda kv: kv[0][0]):
        a_fits, b_fits, diffs = [], [], []
        for _rep, pair in sorted(reps.items()):
            if A in pair and B in pair:
                fa, fb = pair[A]["fit_seconds"], pair[B]["fit_seconds"]
                a_fits.append(fa)
                b_fits.append(fb)
                diffs.append(fa - fb)
        if not diffs:
            continue
        md, sd = mean(diffs), (pstdev(diffs) if len(diffs) > 1 else 0.0)
        n_host = sum(1 for d in diffs if d > 0)
        pct = md / mean(a_fits) if mean(a_fits) else float("nan")
        # Factual signal flag: host edge is "confirmed" only if it beats A in
        # (almost) every rep AND the mean paired diff clears one std (noise band).
        if n_host >= N_REPS - 1 and md > sd:
            signal = "host edge confirmed"
        elif n_host <= 1 and -md > sd:
            signal = "GPU edge"
        else:
            signal = "within noise"
        title = task + (f" K={nc}" if task == "multiclass" else "") + ", 200f"
        lines.append(
            f"| {title} | {mean(a_fits):.2f} | {mean(b_fits):.2f} | "
            f"{md:+.2f} ± {sd:.2f} | {pct:+.1%} | {n_host}/{len(diffs)} | {signal} |"
        )

    lines += ["", "## Per-rep fit times (s)", ""]
    for (task, nc), reps in sorted(by_wl.items(), key=lambda kv: kv[0][0]):
        title = task + (f" K={nc}" if task == "multiclass" else "") + ", 200f"
        lines += [f"### {title}", "",
                  "| rep | A=32768 (GPU) | B=131072 (host) | Δ(A-B) |",
                  "| --- | --- | --- | --- |"]
        for rep, pair in sorted(reps.items()):
            if A in pair and B in pair:
                fa, fb = pair[A]["fit_seconds"], pair[B]["fit_seconds"]
                lines.append(f"| {rep} | {fa:.2f} | {fb:.2f} | {fa - fb:+.2f} |")
        lines.append("")
    with open(REPORT, "w") as fh:
        fh.write("\n".join(lines))
    return lines


def main():
    extract_repo()
    gpu = ensure_cupy()
    rows = run()
    lines = write_report(rows, gpu)
    print("\n".join(lines), flush=True)
    print(f"\nwrote {REPORT}", flush=True)


main()
