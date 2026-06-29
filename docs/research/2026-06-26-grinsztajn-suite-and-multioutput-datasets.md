# Research Note: Grinsztajn 2022 Benchmark Suites and Multi-Output Regression Datasets

**Date:** 2026-06-26
**Author:** literature-scout
**Feeds:** benchmark-overhaul-prompt.md → research-proposer / harness-optimizer

---

## Question

Two deliverables for the benchmark overhaul described in `docs/benchmark-overhaul-prompt.md`:

1. The exact reproducible spec for the Grinsztajn et al. (NeurIPS 2022) tabular benchmark:
   four OpenML suite IDs, all member dataset IDs + names, and the full preprocessing / split /
   HPO protocol.
2. Real multi-target regression datasets on OpenML beyond energy-efficiency (1472) that are
   suitable for the robustness-under-contamination study.

---

## Sources

- Grinsztajn, Oyallon, Varoquaux (2022). "Why do tree-based models still outperform deep
  learning on typical tabular data?" NeurIPS 2022 Datasets & Benchmarks.
  [arXiv:2207.08815](https://arxiv.org/abs/2207.08815) | [NeurIPS proceedings PDF](https://proceedings.neurips.cc/paper_files/paper/2022/file/0378c7692da36807bdec87ab043cdadc-Paper-Datasets_and_Benchmarks.pdf) (2022-12)
- [LeoGrin/tabular-benchmark GitHub repository](https://github.com/LeoGrin/tabular-benchmark)
  (source code for the paper; `src/run_experiment.py` contains the split defaults)
- OpenML API (JSON): queried 2026-06-26 for study IDs 334, 335, 336, 337 and individual
  dataset metadata at `https://www.openml.org/api/v1/json/study/<id>` and
  `https://www.openml.org/api/v1/json/data/<data_id>`
- Müller & Gutjahr (2024). "Better by Default: Strong Pre-Tuned MLPs and Boosted Trees on
  Tabular Data." NeurIPS 2024.
  [arXiv:2407.04491](https://arxiv.org/pdf/2407.04491) — contains secondary description of
  Grinsztajn protocol that clarifies train cap.
- [OpenML tag `2019_multioutput_paper`](https://www.openml.org/api/v1/json/data/list/tag/2019_multioutput_paper)
  — the canonical Mulan multi-target regression dataset collection on OpenML (27 entries;
  queried 2026-06-26).
- Spyromitros-Xioufis et al. (2012/2016). "Multi-target regression via input space expansion."
  Machine Learning. [Springer Link](https://link.springer.com/article/10.1007/s10994-016-5546-z)
  — original paper defining most Mulan MTR datasets.
- [NeurIPS 2022 poster page](https://neurips.cc/virtual/2022/poster/55627) — states
  "a 20,000 compute hours hyperparameter search for each learner."

---

## Deliverable 1 — Grinsztajn 2022 Benchmark: Exact Reproducible Spec

### The Four OpenML Suites

Access programmatically: `openml.study.get_suite(<suite_id>)`.

| Suite ID | Name (OpenML) | Task | Features | Datasets |
|----------|--------------|------|----------|----------|
| **334** | Tabular benchmark categorical classification | Classification | Numerical + categorical | 7 |
| **335** | Tabular benchmark categorical regression | Regression | Numerical + categorical | 17 |
| **336** | Tabular benchmark numerical regression | Regression | Numerical only | 19 |
| **337** | Tabular benchmark numerical classification | Classification | Numerical only | 16 |

Verify live:
- https://www.openml.org/api/v1/json/study/334
- https://www.openml.org/api/v1/json/study/335
- https://www.openml.org/api/v1/json/study/336
- https://www.openml.org/api/v1/json/study/337

### Suite 337 — Numerical Classification (16 datasets)

Member dataset IDs verified via OpenML API (`/api/v1/json/study/337`), names verified
individually via `/api/v1/json/data/<id>`:

| data_id | Name |
|---------|------|
| 44089 | credit |
| 44120 | electricity |
| 44121 | covertype |
| 44122 | pol |
| 44123 | house_16H |
| 44125 | MagicTelescope |
| 44126 | bank-marketing |
| 44128 | MiniBooNE |
| 44129 | Higgs |
| 44130 | eye_movements |
| 45019 | Bioresponse |
| 45020 | default-of-credit-card-clients |
| 45021 | jannis |
| 45022 | Diabetes130US |
| 45026 | heloc |
| 45028 | california |

### Suite 336 — Numerical Regression (19 datasets)

Member dataset IDs verified via OpenML API (`/api/v1/json/study/336`):

| data_id | Name |
|---------|------|
| 44132 | cpu_act |
| 44133 | pol |
| 44134 | elevators |
| 44136 | wine_quality |
| 44137 | Ailerons |
| 44138 | houses |
| 44139 | house_16H |
| 44140 | diamonds |
| 44141 | Brazilian_houses |
| 44142 | Bike_Sharing_Demand |
| 44143 | nyc-taxi-green-dec-2016 |
| 44144 | house_sales |
| 44145 | sulfur |
| 44146 | medical_charges |
| 44147 | MiamiHousing2016 |
| 44148 | superconduct |
| 45032 | yprop_4_1 |
| 45033 | abalone |
| 45034 | delays_zurich_transport |

### Suite 334 — Categorical Classification (7 datasets)

Member dataset IDs verified via OpenML API (`/api/v1/json/study/334`); task IDs also
returned: 361110, 361111, 361113, 361282, 361283, 361285, 361286.

| data_id | Name |
|---------|------|
| 44156 | electricity |
| 44157 | eye_movements |
| 44159 | covertype |
| 45035 | albert |
| 45036 | default-of-credit-card-clients |
| 45038 | road-safety |
| 45039 | compas-two-years |

### Suite 335 — Categorical Regression (17 datasets)

Member dataset IDs verified via OpenML API (`/api/v1/json/study/335`); task IDs also
returned: 361093–361104, 361287–361294.

| data_id | Name |
|---------|------|
| 44055 | analcatdata_supreme |
| 44056 | visualizing_soil |
| 44059 | diamonds |
| 44061 | Mercedes_Benz_Greener_Manufacturing |
| 44062 | Brazilian_houses |
| 44063 | Bike_Sharing_Demand |
| 44065 | nyc-taxi-green-dec-2016 |
| 44066 | house_sales |
| 44068 | particulate-matter-ukair-2017 |
| 44069 | SGEMM_GPU_kernel_performance |
| 45041 | topo_2_1 |
| 45042 | abalone |
| 45043 | seattlecrime6 |
| 45045 | delays_zurich_transport |
| 45046 | Allstate_Claims_Severity |
| 45047 | Airlines_DepDelay_1M |
| 45048 | medical_charges |

### Exact Protocol

**Dataset preprocessing (applied by the authors before upload to OpenML):**
- Missing values removed from the dataset rows.
- Categorical features with **≥ 20 unique categories** are dropped.
- Numerical features with **< 10 unique values** are dropped.
- Numerical features with exactly 2 unique values are converted to categorical.
- The datasets uploaded under these suite IDs have all transformations pre-applied; you do
  not re-apply them when loading.

**Train / validation / test split:**
- `train_prop = 0.70` (70 % of rows for training)
- The remaining 30 % is split evenly into validation (15 %) and test (15 %).
- Training set is capped at **10,000 samples** (medium-scale benchmark); validation and test
  are capped at **50,000 samples** each.
  - "Better by Default" (Müller & Gutjahr 2024) explicitly names these caps in its description
    of the original benchmark protocol.
- No stratification is mentioned for regression; for classification, stratification is
  the conventional choice but is not explicitly stated in the paper.
- Splits use fixed random seeds for reproducibility.

**HPO budget:**
- Random search, **20,000 compute-hours per learner** (stated explicitly in the NeurIPS
  2022 poster abstract: "a 20,000 compute hours hyperparameter search for each learner").
- The repository code (`src/run_experiment.py`) implements an **adaptive `n_iter`** that
  selects how many random configurations to try per dataset based on dataset size:
  - > 6,000 test samples → 1 configuration
  - 3,000–6,000 → 2 configurations
  - 1,000–3,000 → 3 configurations
  - < 1,000 → 5 configurations
- This adaptive schedule is the per-dataset operationalization of the overall compute budget.
- For RepLeafGBM's benchmark overhaul (CPU-only, smaller budget), the practical substitute
  is **30–50 Optuna / random-search trials per model per dataset** with early stopping on
  the validation set — the approach used in subsequent work (Müller & Gutjahr 2024 uses 100
  trials or 23 hours; the RepLeaf overhaul prompt proposes 30–50 trials).

**Regression target transform:**
- No log-transform is applied to regression targets for tree-based models in the original
  benchmark. Target normalization (mean/std) is mentioned only for neural networks in
  the code.

**`fetch_openml` practical gotchas:**
- Always use `data_id=<id>`, not the dataset name, to pin to the correct version
  (OpenML often has multiple versions of similarly-named datasets; the IDs above point to
  the benchmark-specific uploads).
- Use `as_frame=True` so that pandas categorical columns are preserved as such (the default
  changed from False to 'auto' in sklearn 0.24; explicit `as_frame=True` is clearer).
- The target column name varies per dataset. The OpenML task metadata specifies it; the
  easiest approach is to fetch the task (`openml.tasks.get_task(task_id)`) rather than
  the raw dataset, which provides `X, y = task.get_X_and_y(dataset_format='dataframe')`.
- For the suites, iterate over tasks: `suite = openml.study.get_suite(337)`,
  then `for task_id in suite.tasks:`.
- Some Grinsztajn datasets (44143, 44069, 44147) are large (> 100K rows); the 10K training
  cap means they still complete fast, but downloads are large (~1 GB for nyc-taxi).
- Caching: sklearn caches in `~/scikit_learn_data`. Set `data_home=` explicitly if running
  in shared environments to avoid permission issues.
- The `parser="pandas"` default as of sklearn 1.2+ differs slightly from the older
  `"liac-arff"` parser in type inference for ambiguous columns; use `parser="auto"` or
  pin sklearn version for strict reproducibility.

---

## Deliverable 2 — Real Multi-Target Regression Datasets

### Context

The project currently uses only energy-efficiency (OpenML data_id **1472**, 8 features,
2 targets, 768 rows) for multi-output regression. The robustness study in
`experiments/results/2026-06-17-multioutput-real-and-robust.md` showed a decisive
contamination win on this single dataset but needs 3–5 datasets for a significance-tested,
many-dataset claim.

### The Canonical Multi-Target Regression Collection on OpenML

The Mulan multi-target regression (MTR) benchmark sets are tagged `2019_multioutput_paper`
on OpenML. Full list confirmed via API (`/api/v1/json/data/list/tag/2019_multioutput_paper`,
queried 2026-06-26). The table below covers the relevant regression subsets (omits
multi-label classification datasets in the same tag):

| data_id | Name | Rows | Total cols | Confirmed inputs | Confirmed targets |
|---------|------|------|------------|-----------------|-------------------|
| 41474 | andro | 49 | 36 | 30 | 6 |
| 41475 | atp1d | 337 | 417 | 411 | 6 |
| 41476 | atp7d | 296 | 417 | 411 | 6 |
| 41477 | edm | 154 | 18 | 16 | 2 |
| **41478** | **enb** | **768** | **10** | **8** | **2** |
| 41479 | jura | 359 | 18 | 15 | 3 |
| **41480** | **OES10** | **403** | ~314 | ~298 | **16** |
| **41481** | **OES97** | **334** | ~279 | ~263 | **16** |
| 41482 | osales | 639 | 413 | ~399 | ~14 |
| **41483** | **rf1** | **9,125** | 72 | 64 | **8** |
| 41484 | rf2 | 9,125 | 584 | 576 | 8 |
| **41485** | **scm1d** | **9,803** | 296 | 280 | **16** |
| **41486** | **scm20d** | **8,966** | 77 | 61 | **16** |
| 41487 | scpf | 1,137 | 26 | ~23 | ~3 |
| 41488 | sf1 | 323 | 13 | ~10 | 3 |
| **41489** | **sf2** | **1,066** | 13 | ~10 | **3** |
| 41490 | slump | 103 | 10 | 7 | 3 |
| **41491** | **wq** | **1,060** | 30 | 16 | **14** |
| 41492 | youtube | 404 | 31 | ~28 | ~3 |

**Bold** = recommended for the robustness suite (rationale below).

Note: "Total cols" is as reported by the OpenML API; "Confirmed inputs/targets" is from
primary source papers or the OpenML dataset description. "~" means inferred from total
columns minus known targets.

Note on **enb (41478) vs energy-efficiency (1472):** Both are the ENB/UCI energy-efficiency
dataset (8 building parameter inputs, targets = heating load + cooling load, 768 rows). The
tag-41478 version was reformatted for the 2019 multi-output paper. For RepLeafGBM, continue
using data_id **1472** (the canonical OpenML entry already in use in `multioutput_suite.py`).

### Recommended Suite for Robustness Study

Primary recommendation — add these 4 to the existing energy-efficiency (1472):

| data_id | Name | Rows | Inputs | Targets | Rationale |
|---------|------|------|--------|---------|-----------|
| 41491 | wq (water quality) | 1,060 | 16 | 14 | Many targets; good stress test for vector leaves; medium size |
| 41479 | jura | 359 | 15 | 3 | Classic small MTR; heavy metals in soil; different domain |
| 41483 | rf1 (river flow) | 9,125 | 64 | 8 | Large; real environmental data; tests scale |
| 41486 | scm20d (supply chain) | 8,966 | 61 | 16 | Large; many targets; economic domain |

Together with energy-efficiency (1472), this gives 5 datasets spanning:
- Rows: 359 to 9,125 (2.5 decades)
- Inputs: 8 to 64
- Targets: 2 to 16

Secondary candidates (if budget allows expanding to 6–7):
- **41489 sf2** (1,066 rows, ~10 inputs, 3 targets) — solar flare prediction
- **41480 OES10** (403 rows, ~298 inputs, 16 targets) — wide-feature case

Datasets to avoid for the contamination study:
- andro (49 rows), slump (103 rows), edm (154 rows): too few rows to have stable
  contamination signal at 8% (< 10 corrupted rows).
- atp1d/atp7d (411 input features, 296–337 rows): feature count >> row count, different
  regime, would require regularization tuning as a confound.
- rf2 (576 input features): same problem.
- scm1d (296 inputs): dominated by feature dimensionality.

### Fetching Multi-Target Datasets

For these datasets, `fetch_openml(data_id=41491, as_frame=True)` will return a single
default target column (whichever OpenML designates). To get all targets, fetch the raw
data and separate manually. Example pattern:

```python
dataset = openml.datasets.get_dataset(41491)
X_raw, y_raw, _, attr_names = dataset.get_data(dataset_format="dataframe")
# y_raw is None or single-column; X_raw has ALL columns including targets.
# Need to know target column names from the dataset description or metadata.
target_names = dataset.default_target_attribute  # may be a single name
# For multi-output, look at dataset.qualities or dataset description to find all targets.
```

A more robust approach is to check `dataset.features` for columns marked
`is_target=True`. For the wq dataset (41491), the 14 target columns are the species
abundance columns; for jura (41479) they are Cd, Cu, Pb. These are documented in the
original Mulan dataset pages at
[mulan.sourceforge.net/datasets-mtr.html](http://mulan.sourceforge.net/datasets-mtr.html).

---

## Relevance to RepLeafGBM

### (A) Fair-Leaderboard Suite Registry (`benchmarks/suites.py`, to be built)

The Grinsztajn suites are the right anchor for the general leaderboard because:
- They are the benchmark cited in RepLeafGBM's own paper (`docs/paper/repleafgbm-algorithm.tex`)
- They are medium-sized (train ≤ 10K), which is the primary regime RepLeafGBM targets
- They were designed for comparing tree models vs. DL, exactly the comparison the paper needs
- They are recognized by the community (reproduced in TabR, FT-Transformer, Better-by-Default, etc.)

Mapping to code:
- `benchmarks/openml_suite.py` currently has 9 hand-picked datasets. The overhaul would
  replace this with suite-driven loading: `openml.study.get_suite(337)` etc.
- The existing suite uses a 60/20/20 split with `max_rows=6000`; the Grinsztajn protocol
  uses 70/15/15 with train capped at 10K. The overhaul should align to 70/15/15 + 10K cap
  to match the published protocol.
- Suggested default start: suites 336 (numerical regression, 19 datasets) and 337
  (numerical classification, 16 datasets) — 35 datasets, no categorical handling needed.
  Add 334 + 335 once `RepLeafDataset` categorical handling is in scope for the benchmark.
- HPO: use Optuna random search, 50 trials per model per dataset (a practical CPU substitute
  for the original 20,000 compute-hours / adaptive-n_iter scheme). Same validation split
  (15% of data, ≤ 50K rows) for both HPO and early stopping.

### (B) Robust Multi-Output Niche Study

- The recommended 4 datasets (wq, jura, rf1, scm20d) extend the existing
  `benchmarks/multioutput_suite.py` which currently uses only energy-efficiency (1472) +
  synthetic data.
- All 4 are drop-in compatible with `RepLeafRegressor` on 2-D `y` (Phase 22 / v1.5),
  including Huber and quantile objectives (Phase 31).
- The shared-routing vector-leaf architecture (one tree per round, vector output) is tested
  here against per-output independent GBDTs — exactly the architectural claim in the paper.
- Module touch points: `benchmarks/multioutput_suite.py` (add dataset registry entries),
  `experiments/multioutput_real_and_robust.py` (add new datasets to the contamination loop).
- No changes to `src/` are required.

### Guardrail Check

| Concern | Status |
|---------|--------|
| Splitting on embedding dims | Not applicable — datasets are raw tabular; no embedding splitting |
| Updating encoder during boosting | Not applicable — benchmark scripts do not change training code |
| Wrapper-only around LightGBM/XGBoost/CatBoost | Not applicable — the benchmark *compares* against external GBMs, does not wrap them; LightGBM/XGBoost/CatBoost remain in `external/` |
| NumPy path importing torch/external | Not applicable — benchmark scripts are separate from `src/` |

No invariant violations found. These are pure data/protocol findings that feed the harness
design, not the model architecture.

---

## Key Uncertainties and Flags

1. **Suite 335 exact task IDs for `openml.study.get_suite(335)`:** The 17 dataset IDs are
   confirmed but task IDs (needed for `task.get_X_and_y()`) should be verified against the
   live API before use — the study JSON endpoint returns both.

2. **`n_iter="auto"` exact HPO count:** The adaptive schedule from `run_experiment.py` gives
   1–5 random configurations per model per dataset. For a credible benchmark, 50 Optuna
   trials is a stronger and fairer substitute (matches "Better by Default" which used 100
   trials as their proxy). The "20,000 compute-hours" headline is useful for context but
   not directly reproducible on CPU.

3. **osales and scpf target columns:** These datasets from the 2019 multi-output tag are
   not classic Mulan datasets and their target column layout is less well documented.
   Do not include in the primary suite without manual inspection.

4. **OES10 / OES97 feature count:** The API returns total column count; the exact
   input-feature vs. target column split needs to be read from the ARFF file. Both have
   16 targets (confirmed from literature) and ~400 rows. Their feature dimensionality
   (~263–298) is high relative to row count; contamination results may be noisy.

5. **enb (41478) == energy-efficiency (1472):** Strongly suspected to be the same dataset
   (8 features, 768 rows, 2 targets, building energy simulation). Verify with a quick
   correlation check before including both; use only 1472.

---

## Concrete Next Steps

1. **For harness-optimizer:** Build `benchmarks/suites.py` as a registry with:
   - `GRINSZTAJN_SUITES = {336: "num_reg", 337: "num_cls", 335: "cat_reg", 334: "cat_cls"}`
   - A `load_suite(suite_id, max_train=10000, train_prop=0.70, seed=42)` function that
     calls `openml.study.get_suite()`, iterates over task IDs, calls
     `task.get_X_and_y(dataset_format="dataframe")`, and applies the 70/15/15 split.
   - Start with suites 336 + 337 (numerical only, 35 datasets) for the first pass.

2. **For harness-optimizer (multi-output):** Extend `benchmarks/multioutput_suite.py`
   with a `MULAN_SUITE` registry adding data_ids 41491 (wq), 41479 (jura), 41483 (rf1),
   41486 (scm20d). Add a `load_multioutput_dataset(data_id)` helper that fetches the ARFF,
   identifies target columns from the dataset metadata, and returns `(X, Y)` arrays.

3. **For research-proposer:** The Grinsztajn protocol (70/15/15, 10K cap, 50 Optuna
   trials, mean rank + Wilcoxon signed-rank + CD diagram as statistics) is a complete,
   citable spec. Use it directly for the fair-leaderboard study design rather than
   designing ad-hoc splits.
