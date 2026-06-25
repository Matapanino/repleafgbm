"""Sizing reads for the Colab T4 session (iter-010 + Task-B decisions).

Runs embedded_linear cuda fits with REPLEAFGBM_PROFILE=1 (batched depthwise scan
default ON) and dumps the full per-phase breakdown, so the loop can decide WITHOUT
building blind:
  - iter-010 (batched build_histograms): the histogram vs split_scan share on deep
    depthwise. Build only if histogram is a meaningful share of fit (the gate is
    ~>=10%); sibling-subtraction already halves the per-level build launches.
  - Task B (leafwise frontier-batch): the split_scan share of a LEAFWISE cuda fit —
    that share (x the ~2x M=2 amortization ceiling) is B's true prize, currently
    unmeasured on the default grow_policy.
Writes /content/sizing_report.md. Run on the GPU VM (uploaded with the working tree).
"""
import os
import sys
import time

import numpy as np

for _repo in ("/content/repleafgbm", os.getcwd()):
    if os.path.isdir(os.path.join(_repo, "src", "repleafgbm")):
        os.chdir(_repo)
        break
sys.path.insert(0, "src")
os.environ["REPLEAFGBM_PROFILE"] = "1"
os.environ.pop("REPLEAFGBM_CUDA_BATCHED_SCAN", None)  # use the new default (ON)

# (name, grow_policy, task, n, n_features, n_classes, depth, n_estimators)
CASES = [
    ("depthwise-wide", "depthwise", "reg", 50_000, 200, 1, 8, 25),
    ("depthwise-mc5", "depthwise", "clf", 50_000, 200, 5, 8, 20),
    ("leafwise-wide", "leafwise", "reg", 50_000, 200, 1, 0, 25),  # Task-B ceiling
]
REPS = 3


def _data(task, n, f, n_classes, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, f))
    sig = 2 * X[:, 0] + np.sin(X[:, 1]) - 1.5 * (X[:, 2] > 0)
    if task == "clf":
        edges = np.quantile(sig, np.linspace(0, 1, n_classes + 1)[1:-1])
        return X, np.digitize(sig, edges)
    return X, sig + rng.normal(scale=0.1, size=n)


def run(name, gp, task, n, f, n_classes, depth, n_est):
    from repleafgbm import RepLeafClassifier, RepLeafRegressor
    X, y = _data(task, n, f, n_classes)
    Est = RepLeafRegressor if task == "reg" else RepLeafClassifier
    kw = dict(grow_policy=gp, num_leaves=63, n_estimators=n_est,
              leaf_model="embedded_linear", encoder="identity",
              max_leaf_emb_dim=256, split_backend="cuda", random_state=0)
    if gp != "leafwise":
        kw["max_depth"] = depth
    Est(**kw).fit(X, y)  # warmup
    dts, ps_list = [], []
    for _ in range(REPS):
        m = Est(**kw)
        t = time.perf_counter()
        m.fit(X, y)
        dts.append(time.perf_counter() - t)
        ps_list.append(dict(getattr(m, "phase_seconds_", {})))
    keys = set().union(*ps_list)
    ps = {k: float(np.median([p.get(k, 0.0) for p in ps_list])) for k in keys}
    fit = sum(v for k, v in ps.items() if k != "predict")
    return name, gp, float(np.median(dts)), fit, ps


def main():
    rows = [run(*c) for c in CASES]
    lines = [
        "# Colab T4 sizing — iter-010 (batched histogram) + Task-B (leafwise) ",
        "",
        "embedded_linear, split_backend=cuda, batched depthwise scan default ON, "
        f"median of {REPS}, REPLEAFGBM_PROFILE.",
        "",
        "| case | policy | fit(s) | histogram | split_scan | leaf_fit | hist% | scan% |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, gp, _dt, fit, ps in rows:
        h, s, ll = ps.get("histogram", 0.0), ps.get("split_scan", 0.0), ps.get("leaf_fit", 0.0)
        hp, sp = (100 * h / fit, 100 * s / fit) if fit else (0.0, 0.0)
        lines.append(f"| {name} | {gp} | {fit:.2f} | {h:.3f} | {s:.3f} | {ll:.3f} "
                     f"| {hp:.1f}% | {sp:.1f}% |")
        print(f"{name:16s} fit={fit:6.2f}s  hist%={hp:5.1f}  scan%={sp:5.1f}  "
              f"leaf%={100*ll/fit if fit else 0:5.1f}")
    open("/content/sizing_report.md", "w").write("\n".join(lines) + "\n")
    print("wrote /content/sizing_report.md")


if __name__ == "__main__":
    main()
