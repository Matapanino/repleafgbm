# Proposal: Per-leaf adaptive leaf model (`leaf_model="adaptive"`)

- **Status:** Implemented; **KEEP_EXPERIMENTAL** after the validating study (see
  [§12 Outcome](#12-outcome-post-experiment-2026-06-23-keep_experimental)). This document
  formalizes the approved plan into the proposal/ADR spec format. It does **not** redesign
  it — where this spec and the plan disagree, the plan wins.
- **Date:** 2026-06-23
- **Author:** research-proposer
- **Type:** Algorithm-feature addition (new leaf-model variant + data-driven per-leaf
  model selection). **Not** a performance optimization; **not** a default change.
- **Thesis check:** PASS — no thesis/architecture violation (see
  [Guardrail check](#7-guardrail--thesis-check)). Routing stays raw-feature-only, leaves
  stay representation-conditioned, encoder stays frozen, no embedding-dim splits, no
  `FORMAT_VERSION` bump, no native rebuild.
- **Companion ADR:** to be drafted alongside implementation as
  `docs/adr/<next>-adaptive-leaf-model.md`.

---

## 1. Problem

The leaf-model choice is currently global and the evidence says the *right* choice is
**dataset- and even leaf-dependent**:

- **Regression is solved by `embedded_linear`.** After the Phase-7 z-clip extrapolation
  guard, embedded-linear beats constant on **3/3** real regression datasets and beats
  LightGBM-native on 2/3 (`experiments/results/real_data_validation.md` Phase 7).
- **Binary is the opposite quadrant.** Constant beats embedded-linear on **4/4** binary
  datasets in the Phase-25 OpenML suite (`experiments/results/openml_benchmark.md`).
  The Phase-12 mechanism is diagnosed (`experiments/results/binary_leaf_gain.md`): the
  logistic Hessian `h = p(1-p)` collapses in confident regions, and late leaf-linear
  fits keep `||w|| ≈ 0.45` with **no out-of-sample signal** — pure noise absorption. The
  tell is the weight-norm trajectory: regression `||w||` collapses `0.75 → 0.09` as
  recoverable signal runs out, while binary `||w||` *stays* ~`0.45` through 100 rounds.
- **Global reweighting cannot fix it.** Every global remedy was rejected on real data:
  the `l2` sweep is within seed noise, the Hessian floor is mildly *harmful* (it distorts
  `t = -g/h` in exactly the confident rows), and damped `h^α` is a wash
  (`binary_leaf_gain.md` §2). So the actionable lever is per-leaf model **selection**, not
  reweighting.

A single global `leaf_model` flag forces a per-dataset choice the library cannot make for
the user, and even within one dataset cannot exploit that *some* leaves carry recoverable
linear structure while others (confident, low-ESS) carry only noise. External grounding:
GBDT-PL (arXiv:1802.05640) and LightGBM `linear_tree` show piecewise-linear leaves help
**when well-conditioned**; Kish ESS / regression leverage are the standard signals for
"is this fit trustworthy out-of-sample?". The design below gives the linear-leaf gain
where the fit is well-conditioned and falls back to constant where it is not — decided
per leaf, from the data, at no native cost.

**Outcome.** A new opt-in `leaf_model="adaptive"` that selects constant↔embedded_linear
**per leaf** via a closed-form weighted-ridge leave-one-out (LOO) gate. Existing defaults
are **unchanged** (`leaf_model` stays `"embedded_linear"`). Promotion to default is a
separate `results-analyst`-gated decision after the real-data study (§9).

---

## 2. Hypothesis (why it could help *this* architecture)

The thesis pushes capacity into the representation-conditioned leaf `b + w·z_θ(x)`. The
*linear* part is exactly where the binary failure lives: a leaf with heterogeneous,
mostly-tiny `hᵢ` (confident rows) has high per-row leverage, so the in-sample residual is
driven near zero by a `w` that does not generalize. Constant leaves dodge this; linear
leaves win when the leaf has genuine, well-conditioned smooth structure.

Hypothesis: a **per-leaf, leverage-corrected** estimate of out-of-sample leaf error can
separate these two regimes from the data alone — keeping the embedded-linear gain on the
regression-like leaves and demoting the noise-absorbing leaves to constant — and thereby
match `embedded_linear` on regression *and* `constant` on binary without a global flag.
This exploits a degree of freedom the frozen-encoder architecture currently leaves on the
table: the leaf-model **form** is a free, per-leaf choice that the existing global flag
collapses.

---

## 3. The gate (closed-form weighted-ridge LOO)

The leaf objective is the existing one (`docs/math.md`,
`leaf_models.py::_fit_weighted_ridge`):
`½ Σᵢ hᵢ (b + w·zᵢ − tᵢ)² + ½ λ ‖w‖²`, with `tᵢ = −gᵢ/hᵢ`.
Per **linear-eligible** leaf (those already passing the existing size pre-filter
`min_n = max(min_samples_linear, emb_dim + 2)`), with `h_sum = Σ hᵢ`, centered means
`z̄ = s_hz / h_sum`, `t̄ = −g_sum / h_sum`, centered rows `z̃ᵢ = zᵢ − z̄`, and the solved
centered Gram `M = A_c + λI` (the same `A`/`M` the batched solve already builds):

- **leverage** `Hᵢᵢ = hᵢ · (z̃ᵢᵀ M⁻¹ z̃ᵢ) + hᵢ / h_sum` — the weighted-ridge hat-matrix
  diagonal (the `hᵢ / h_sum` term is the intercept's contribution from centering). Clamp
  `Hᵢᵢ ≤ 1 − 1e-6` for numerical safety.
- **residual** `rᵢ = w·z̃ᵢ + (t̄ − tᵢ)` (the leaf's in-sample residual at its fitted
  `(b, w)`, written in centered coordinates).
- **LOO errors**
  - `E_lin   = Σ hᵢ ( rᵢ / (1 − Hᵢᵢ) )²`  — weighted PRESS for the linear fit.
  - `E_const = Σ hᵢ ( (t̄ − tᵢ) / (1 − hᵢ/h_sum) )²` — weighted PRESS for the
    constant-only fit (intercept-only hat diagonal is `hᵢ/h_sum`).
- **verdict:** keep the linear fit **iff** `E_lin < (1 − μ) · E_const`, with
  `μ = leaf_gate_margin` (default `0.01`). Comparison is **strict `<`**; an exact tie or
  any non-finite `E_lin`/`E_const`/`Hᵢᵢ` ⇒ **constant** (deterministic, conservative).
- **Demotion is the existing constant fallback.** A leaf the gate sends to constant is
  written with the *already-supported* zero-weights + `±inf` z-clip encoding — i.e. the
  `else` branch of `_solve_and_assemble` (`leaf_models.py:357–359`). No new leaf encoding.

**Naive baseline arm** `leaf_gate="insample"`: identical rule but with
`E_lin_insample = Σ hᵢ rᵢ²` (drop the leverage correction, i.e. `Hᵢᵢ ≡ 0`). This arm
exists **only** to demonstrate that the LOO correction beats the cheap in-sample signal;
it is not a recommended setting.

### 3.1 Why LOO, not aggregate GCV, not an eval-set holdout (verified rationale)

- **vs GCV.** The binary failure is precisely where **per-row leverage is heterogeneous**
  (tiny `hᵢ` confident rows → high `Hᵢᵢ`). GCV replaces every `Hᵢᵢ` with the mean
  leverage `tr(H)/n`, which *averages that signal away* — exactly the rows that make the
  linear fit untrustworthy get smeared into the bulk. LOO keeps the per-row term, so it
  can see the few high-leverage rows that drive the noise absorption.
- **vs eval-set holdout.** A holdout gate needs a validation split threaded through
  **every** fit path (`fit_leaves`, `fit_leaves_multiclass`, `fit_vector_leaves`) and
  **three** booster loops, degrades on sparse / no-eval leaves (small leaves get an empty
  or tiny holdout — the worst case is also where the decision matters most), and adds a
  new determinism surface (the split RNG). The closed-form LOO reuses statistics the
  solve already materializes and adds only a per-row pass over each leaf — no signature
  change, no new RNG.

---

## 4. Design

- **New additive variant.** `AdaptiveLeafModel(EmbeddedLinearLeafModel)` registered under
  `leaf_model="adaptive"`. It reuses the entire embedded-linear fit (batched Gram + solve)
  and inserts the gate as a post-solve verdict, demoting gated-out leaves via the existing
  constant-fallback branch. **Subclass, not a rewrite** — the linear fit is identical, so
  a kept-linear adaptive leaf is bitwise the embedded-linear leaf.
- **Defaults unchanged.** `leaf_model` stays `"embedded_linear"`; `leaf_gate_margin` and
  `leaf_gate` are inert unless `leaf_model="adaptive"`. Promotion to default is out of
  scope here (separate `results-analyst` gate after §9).
- **HOST-SIDE ONLY.** The gate consumes already-materialized per-leaf statistics (`M`,
  `s_hz`, `g_sum`, `h_sum`, `z_mean`, `t_mean`, the solved `w`) plus a Python per-row pass
  per leaf. **No Rust/native change**, no `BaseSplitBackend` change, no `leaf_linear_stats`
  change. (`M⁻¹` is applied via a batched solve / triangular solve against `z̃`, not an
  explicit inverse.)
- **No serialization bump.** `FORMAT_VERSION` stays **6**. A gated-to-constant leaf is the
  already-round-tripping zero-weights + `±inf`-clip encoding; mixed-form trees (some
  constant, some linear) already serialize and predict today. Nothing new is written.
- **Reproducibility contract.** Same backend + same seed ⇒ **deterministic / bitwise**
  (no RNG added; the gate is a pure function of the fitted statistics). **Cross-backend**
  (NumPy vs Rust) the gate is *allclose-not-bitwise*: it is a thresholded comparison and
  the `μ` dead-band (`≥ 1%` of `E_const`) is far larger than the ~`1e-6` cross-backend
  statistical noise, so the **selected linear-leaf SET is identical** away from a
  near-boundary leaf. Documented as a known property (mirrors the CUDA allclose contract).

### 4.1 One-verdict-per-leaf for vector / multiclass leaves

- **Multi-output** (`fit_vector_leaves`): the routing and the weighted Gram are shared
  across outputs (squared-error Hessian identical per column, see `multioutput.py:82–94`),
  so the **leverage `Hᵢᵢ` is shared** across outputs and the gate sums `E_lin`/`E_const`
  **over outputs** to a single per-leaf verdict (a vector leaf is kept-linear or demoted
  as a whole — its `weights[i]` block is zeroed and z-clip disabled on demotion).
- **Multiclass** needs **no special code**: the pooled multiclass path funnels every
  `(class, leaf)` through `_solve_and_assemble`, so each `(class, leaf)` gets its own gate
  verdict for free.

---

## 5. Code-map touch points (exact)

| File / symbol | Change |
|---|---|
| `core/leaf_models.py` (new) | `AdaptiveLeafModel(EmbeddedLinearLeafModel)`; a **shared free helper** computing the LOO verdict from `(M, z̃-stats, w, g_sum, h_sum, z_mean, t_mean, μ, gate_mode)` → boolean keep-mask over `linear`. |
| `core/leaf_models.py::_solve_and_assemble` (~`353–359`) | **Gate seam**: after the batched solve, before the per-leaf write loop, compute the keep-mask via the helper; for gated-out leaves take the **existing constant-fallback branch** (`357–359`: keep Newton `bias`, set `z_min/z_max = ±inf`, leave `weights[i]` zero) instead of writing `w[j]`. Subclass overrides only the seam (e.g. via a hook on the keep-mask), keeping the base path byte-identical when not adaptive. |
| `core/leaf_models.py::make_leaf_model` (`474–486`) | add the `"adaptive"` branch returning `AdaptiveLeafModel(...)`; thread **keyword-only** `leaf_gate_margin` / `leaf_gate` (with `embedded_linear`/`constant`/`raw_linear` ignoring them). Update the `ValueError` available-list. |
| `core/multioutput.py::fit_vector_leaves` (`38–106`) | apply the **one-verdict-per-leaf** gate (shared leverage, `E` summed over outputs) using the same shared helper; read `getattr(leaf_model, "leaf_gate_margin", None)` and `getattr(leaf_model, "leaf_gate", "loo")` (margin `None` ⇒ gate **off** — preserves current behavior for non-adaptive models). |
| `core/multiclass.py` | **no change** (pooled path already funnels `(class, leaf)` through `_solve_and_assemble`). |
| `sklearn.py::BaseRepLeafModel.__init__` (`180–225`) | add `leaf_gate_margin: float = 0.01` and `leaf_gate: str = "loo"`; set `self.leaf_gate_margin` / `self.leaf_gate`. (sklearn introspection ⇒ they auto-serialize in `model_config.json` and round-trip; **no format bump**.) |
| `sklearn.py::BaseRepLeafModel.fit` (`269–271`) | validate `leaf_gate_margin ≥ 0` (else `ValueError`) and `leaf_gate ∈ {"loo","insample"}`; pass both into `make_leaf_model(...)`. Defaults unchanged ⇒ inert unless `leaf_model="adaptive"`. |
| `regressor.py`, `classifier.py` | **no change** (inherit base `__init__`; `_make_booster` is leaf-model-agnostic). |
| `core/serialization.py` | **no change** (`FORMAT_VERSION` stays **6**). |
| `backends/`, `native/`, `backends/cuda_backend.py` | **no change** (host-side gate; no maturin rebuild; crate version not bumped). |

---

## 6. Required tests (`tests/test_leaf_models.py`)

Match existing style (`np.testing.assert_allclose`; structure via `model.booster_`):

- **Recovers constant on noise.** A planted leaf whose `t` is pure noise vs `z` ⇒ gate
  demotes to constant (zero weights, `±inf` clip).
- **Recovers linear on clean signal.** A planted leaf with `t = w*·z + small_noise` ⇒
  gate keeps the linear fit (`||w|| > 0`).
- **Mixed verdicts in one tree.** A single tree fit on data with both leaf types carries a
  mix of constant-demoted and kept-linear leaves.
- **Demotion is bitwise the existing fallback.** A constant-gated leaf's `LeafValues`
  (`bias`, zero `weights`, `±inf` `z_min/z_max`) equals `EmbeddedLinearLeafModel`'s
  constant-fallback for the same leaf **bitwise**.
- **Margin brackets.** `leaf_gate_margin = 0` ⇒ keep-linear whenever
  `E_lin < E_const` (most permissive); a large `μ` (e.g. `1e9`) ⇒ **all** leaves demote to
  constant. Brackets the all-linear / all-constant extremes.
- **Save/load round-trip.** A mixed adaptive model `save_model`/`load_model` predicts
  identically (`allclose`); assert `model_config.json["format_version"] == 6` and that
  `leaf_gate_margin` / `leaf_gate` round-trip via `get_params()`.
- **Cross-backend gate stability** (`test_adaptive_gate_stable_across_backends`): on a
  fixture with **no near-boundary leaf**, the *selected linear-leaf SET* (which leaves
  kept `w`) is identical NumPy vs Rust; predictions `allclose`.
- **Determinism.** Same seed ⇒ same model (two independent fits' predictions
  `assert_allclose`), each task.
- **`insample` arm.** `leaf_gate="insample"` parses, fits, and (on a planted high-leverage
  leaf) keeps a linear fit that `leaf_gate="loo"` demotes — i.e. the LOO correction
  measurably changes the verdict.
- **Multi-output one-verdict.** `fit_vector_leaves` with `leaf_model="adaptive"` produces
  vector leaves that are kept/demoted **as a whole** (no per-output split of the verdict).

---

## 7. Guardrail / thesis check

Independent sanity-check against the project invariants. **Result: PASS — no violation.**

| Invariant | Status | Why |
|---|---|---|
| **Raw-feature routing only** | PASS | The gate runs entirely **inside leaf fitting**; it never touches splits. Routing is unchanged. The feature only picks between **two existing leaf forms** on an already-grown partition. |
| **Leaf-only representation** | PASS | `z_θ(x)` is used exactly as today (the embedded-linear fit). The gate adds a verdict, not a new use of `Z`. |
| **Encoder frozen during boosting** | PASS | No θ update path is added; `freeze_encoder` guard (`sklearn.py:250–254`) unchanged. The gate consumes embeddings fetched once before the grow loop. |
| **No embedding-dim splits** | PASS | No split code changes; the gate indexes leaf rows, never proposes a split on a `z`-dim. |
| **Newton-target leaf fitting + extrapolation guard** | PASS | Leaves still fit `t = −g/h` weighted by `h`; kept-linear leaves keep the per-leaf `z_min/z_max` clip; demoted leaves use the existing `±inf` constant-fallback (clip disabled, as today). |
| **No wrapper-only-around-LightGBM/XGBoost** | PASS | Native path; nothing imports `external`/torch/lightgbm/cupy. |
| **Determinism** (`check_random_state`; same seed ⇒ same model) | PASS | The gate is a pure function of fitted statistics; **no RNG added**. Same backend + seed is bitwise; the conservative tie/non-finite → constant rule removes ambiguity. |
| **NumPy⇄Rust parity** | PASS | **No native change.** The gate is host-side over backend-agnostic statistics; the linear fit and histograms are untouched (bitwise NumPy⇄Rust as before). Cross-backend the gate is *allclose-not-bitwise* by construction (`μ` dead-band ≫ `1e-6`), documented. |
| **SemVer / serialization read-ladder** | PASS | `leaf_gate_margin` / `leaf_gate` are additive, defaulted params (minor-version feature); `FORMAT_VERSION` stays **6**; old models still read; a demoted leaf is the already-supported encoding ⇒ **no new format**. |
| **Optional-deps isolation** | PASS | No native/optional import touched; crate version not bumped. |

---

## 8. Risks

- **Leverage formula / numerical edge cases.** `M⁻¹` ill-conditioning or `Hᵢᵢ → 1` could
  blow up the PRESS. *Mitigation:* clamp `Hᵢᵢ ≤ 1 − 1e-6`; non-finite `E_lin`/`E_const` ⇒
  constant (conservative); apply `M⁻¹` via solve, never an explicit inverse; the
  "recovers linear on clean signal" + "recovers constant on noise" tests pin both ends.
- **Near-boundary cross-backend flips.** A leaf with `E_lin ≈ (1−μ)·E_const` could flip
  verdict between backends. *Mitigation:* documented allclose-not-bitwise contract; the
  cross-backend stability test uses a **no-near-boundary** fixture; same-backend stays
  bitwise.
- **Subclass seam leaking into the base path.** The gate hook must not perturb
  non-adaptive `_solve_and_assemble` output. *Mitigation:* base path byte-identical when
  the keep-mask is "all true" (margin `None`/off); the existing leaf-model suite stays
  green with no drift (regression guard); demotion == existing fallback **bitwise** test.
- **Per-row pass cost.** The gate adds an `O(n_leaf · emb_dim)` pass per linear leaf.
  *Mitigation:* host-side, vectorizable per leaf, gated on already-eligible leaves only;
  feasibility-of-perf is a `native-optimizer` question only if a profile flags it (not
  expected at v0 scale).
- **Null result.** The gate may not beat the better of {constant, embedded_linear} on any
  real dataset. *Mitigation:* ship opt-in, default unchanged, record the measured gate
  behavior honestly (a null result is publishable here — CLAUDE.md honesty rule).

---

## 9. Validating experiment

After green + `core-reviewer` sign-off, hand to **experiment-runner** →
**results-analyst**. Harnesses: `benchmarks/benchmark_real_data.py` and
`benchmarks/openml_suite.py`.

- **Arms:** `constant`, `embedded_linear`, **`adaptive` (LOO)**, **`adaptive_insample`**
  (naive arm to show LOO beats the cheap signal).
- **Config:** `encoder="identity"`, capacity-matched across arms, **≥5 seeds**
  (mean ± std; single-seed deltas treated as noise per memory), `leaf_gate_margin` sweep
  `{0, 0.01, 0.05}`.
- **Data/tasks:** regression + binary + multiclass + categorical (real datasets);
  synthetic only as reference.
- **Metric:** task-appropriate (RMSE / logloss / accuracy / multiclass logloss); primary
  read is per-dataset rank of the four arms.
- **HONEST win bar** (tie ≠ win):
  - `adaptive ≥ embedded_linear − 1σ` on **all** regression datasets, **and**
  - `adaptive ≥ constant − 1σ` on binary (adult), **and**
  - `adaptive ≥ best-current − 1σ` on multiclass.
- **Hypothesis / expected effect:** `adaptive` tracks `embedded_linear` on regression
  (most leaves kept-linear) and tracks `constant` on binary (confident, low-ESS leaves
  demoted); `adaptive_insample` over-keeps linear on binary (no leverage correction) and
  thus underperforms `adaptive` there — the demonstration that the LOO correction is what
  matters.
- **Verdict (results-analyst):** whether `adaptive` clears the win bar and whether the
  default should change. **Default change only with an evidence-backed report.** If it
  ties everywhere, **still ship it opt-in**, default unchanged, and record the measured
  gate behavior plainly (null path).

---

## 10. Recommendation

**Proceed.** A faithful, thesis-preserving algorithm-feature addition with a contained
blast radius (host-side gate, subclass of the existing linear leaf, demotion via the
already-supported constant-fallback encoding, no format bump, no native rebuild) and a
clear win bar. Sequence: (1) add `AdaptiveLeafModel` + the shared LOO helper + the
`_solve_and_assemble` gate seam and confirm the existing leaf-model suite stays green with
no drift (base-path regression guard); (2) wire the `fit_vector_leaves` one-verdict path;
(3) plumb `leaf_gate_margin` / `leaf_gate` through `make_leaf_model` + sklearn; (4) tests
(§6); (5) `qa-verifier` green gate; (6) `core-reviewer` sign-off (thesis / parity /
determinism / serialization / subclass seam); (7) experiment-runner → results-analyst for
the verdict. **Keep `leaf_model="embedded_linear"` the default absent contrary evidence.**

---

## 11. References

- Evidence: `experiments/results/real_data_validation.md` (Phase 7 regression 3/3),
  `experiments/results/openml_benchmark.md` (Phase 25 binary 4/4),
  `experiments/results/binary_leaf_gain.md` (Phase 12 mechanism + rejected reweighting).
- Math: `docs/math.md` (Newton targets, weighted-ridge leaf objective, centering).
- Code: `core/leaf_models.py` (`_solve_and_assemble`, `make_leaf_model`,
  `EmbeddedLinearLeafModel`), `core/multioutput.py` (`fit_vector_leaves`), `sklearn.py`
  (`BaseRepLeafModel`), `core/serialization.py` (`FORMAT_VERSION = 6`).
- External: GBDT-PL (arXiv:1802.05640), LightGBM `linear_tree`; Kish effective sample
  size / regression leverage (PRESS) for the trustworthiness signal.
- Companion ADR (to be drafted): `docs/adr/<next>-adaptive-leaf-model.md`.

---

## 12. Outcome (post-experiment, 2026-06-23): KEEP_EXPERIMENTAL

The validating study (§9) ran at **5 seeds** on `benchmarks/benchmark_real_data.py`
(california, house_sales, diamonds, adult) and `benchmarks/openml_suite.py` (9 datasets).
Reports: `experiments/results/2026-06-23-adaptive-leaf-gating-realdata.md` and
`…-openml.md`.

**Result — within seed noise.** On the four real datasets that carry per-seed dispersion,
`adaptive` tracked the *better* of `constant` and `embedded_linear` and was never the worst
of the three — but **every** adaptive-vs-baseline difference was well inside one standard
deviation. There were **no decisive ≥1σ wins (or losses)**. On adult (the binary
noise-absorption case) `adaptive` sat between `constant` and `embedded_linear` within noise
and beat `adaptive_insample`, consistent with the leverage correction mattering most there;
on the easier datasets `adaptive_insample` was within noise of `adaptive`. The OpenML suite
(point means only — direction, not significance) ranked `adaptive` the best RepLeaf arm on
both regression and classification mean rank, but with no per-seed std this cannot be
significance-tested.

**Verdict: KEEP_EXPERIMENTAL.** Ship `adaptive` as an **opt-in, experimental** leaf model
and **keep the default unchanged** (`embedded_linear`). The evidence supports a *robust
per-leaf hedge* between constant and embedded-linear ridge leaves — it removes the manual
per-task `constant`-vs-`embedded_linear` choice and is no-worse-within-noise — **not** a
general, statistically separated accuracy improvement. No default change is warranted on
this evidence.

**Scope clarification.** This is **per-leaf model *selection*** between a constant leaf and
the existing **embedded-linear ridge** leaf. It is **not** jointly-trained leaf embeddings
and not a new representation: the encoder stays frozen and `Z` is unchanged.
`leaf_gate="insample"` remains a **diagnostic baseline, not a recommended setting**.

Deferred follow-ups: a companion ADR if/when a default change is reconsidered; learned
encoders (`encoder="torch_periodic_plr"`/`"torch_plr"`) as adaptive *benchmark* arms (no
API change).
