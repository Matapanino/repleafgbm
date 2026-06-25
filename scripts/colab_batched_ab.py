"""Interleaved A/B: depthwise cuda fit with node-batched scan OFF vs ON (Colab T4).

Measures whether REPLEAFGBM_CUDA_BATCHED_SCAN (the depthwise level-batched device
scan, Stage 2) beats the per-node device scan on the split_scan phase + whole fit,
and confirms quality-equivalence. Deep depthwise trees give many nodes per level
(the batching unit); the win grows with that node count. Run on the GPU VM
(uploaded with the working tree); writes /content/batched_ab_report.md.
"""
import os
import sys
import time

import numpy as np

# Run from the working tree the GPU loop extracted (colab_remote_test.py puts it
# at /content/repleafgbm); fall back to cwd for local use.
for _repo in ("/content/repleafgbm", os.getcwd()):
    if os.path.isdir(os.path.join(_repo, "src", "repleafgbm")):
        os.chdir(_repo)
        break
sys.path.insert(0, "src")

REPS = 5
# (name, n_rows, n_features, max_depth, n_estimators)
CASES = [
    ("wide", 50_000, 200, 8, 25),
    ("narrow", 100_000, 30, 8, 25),
    ("multiclass", 50_000, 200, 8, 20),
]


def _data(name, n, f, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, f))
    signal = 2 * X[:, 0] + np.sin(X[:, 1]) - 1.5 * (X[:, 2] > 0)
    if name == "multiclass":
        edges = np.quantile(signal, [0.33, 0.66])
        y = np.digitize(signal, edges)
        return X, y, "clf"
    y = signal + rng.normal(scale=0.1, size=n)
    return X, y, "reg"


def _fit(X, y, kind, batched, depth, n_est):
    os.environ["REPLEAFGBM_CUDA_BATCHED_SCAN"] = "1" if batched else "0"
    os.environ["REPLEAFGBM_PROFILE"] = "1"
    from repleafgbm import RepLeafClassifier, RepLeafRegressor
    Est = RepLeafRegressor if kind == "reg" else RepLeafClassifier
    m = Est(grow_policy="depthwise", max_depth=depth, num_leaves=100_000,
            n_estimators=n_est, leaf_model="constant", split_backend="cuda",
            random_state=0)
    t = time.perf_counter()
    m.fit(X, y)
    dt = time.perf_counter() - t
    return dt, dict(getattr(m, "phase_seconds_", {})), m


def _quality(m, X, y, kind):
    from sklearn.metrics import accuracy_score, r2_score
    if kind == "reg":
        return float(r2_score(y, m.predict(X)))
    return float(accuracy_score(y, m.predict(X)))


def main():
    rows = []
    for name, n, f, depth, n_est in CASES:
        X, y, kind = _data(name, n, f)
        _fit(X, y, kind, False, depth, n_est)  # warmup
        fa, fb, sa, sb = [], [], [], []
        qa = qb = None
        for rep in range(REPS):
            order = [False, True] if rep % 2 == 0 else [True, False]
            res = {}
            for batched in order:
                dt, ps, m = _fit(X, y, kind, batched, depth, n_est)
                res[batched] = (dt, ps.get("split_scan", 0.0), _quality(m, X, y, kind))
            fa.append(res[False][0])
            fb.append(res[True][0])
            sa.append(res[False][1])
            sb.append(res[True][1])
            qa, qb = res[False][2], res[True][2]
        med = lambda v: float(np.median(v))  # noqa: E731
        r = dict(
            case=name, shape=f"{n}x{f}", depth=depth,
            fit_off=med(fa), fit_on=med(fb), fit_x=med(fa) / med(fb),
            scan_off=med(sa), scan_on=med(sb),
            scan_x=(med(sa) / med(sb) if med(sb) > 0 else 0.0),
            q_off=qa, q_on=qb, q_absdiff=abs(qa - qb),
        )
        rows.append(r)
        print(r)

    lines = [
        "# Node-batched depthwise scan A/B (Colab T4)", "",
        "REPLEAFGBM_CUDA_BATCHED_SCAN off (per-node device scan) vs on "
        "(level-batched). `scan` = split_scan phase seconds (median of 5, "
        "interleaved). Quality: r2 (reg) / accuracy (clf).", "",
        "| case | shape | depth | fit off | fit on | fit× | scan off | scan on "
        "| scan× | |Δq| |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['case']} | {r['shape']} | {r['depth']} | {r['fit_off']:.2f} | "
            f"{r['fit_on']:.2f} | **{r['fit_x']:.2f}x** | {r['scan_off']:.3f} | "
            f"{r['scan_on']:.3f} | **{r['scan_x']:.2f}x** | {r['q_absdiff']:.1e} |"
        )
    open("/content/batched_ab_report.md", "w").write("\n".join(lines) + "\n")
    print("wrote /content/batched_ab_report.md")


if __name__ == "__main__":
    main()
