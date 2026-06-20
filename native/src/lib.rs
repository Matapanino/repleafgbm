//! Rust split-search kernels for RepLeafGBM.
//!
//! Implements the two functions of the `BaseSplitBackend` contract
//! (docs/backend_strategy.md, Axis 1). Semantics mirror the NumPy reference
//! kernel in `backends/numpy_backend.py`, including tie-breaking (first
//! maximum in feature-major, bin-minor order; stable category sort) and
//! floating-point accumulation order where it is observable (per-bin
//! accumulation in row order; cumulative sums with the missing block added
//! per candidate), so the two backends agree to numerical noise.
//!
//! `build_histograms` is parallelized across *features* (rayon): each feature
//! owns a disjoint output slice and accumulates its bins in row order, so every
//! `(feature, bin)` cell sees the exact same summation order as the serial scan
//! — the histograms stay **bitwise-identical** to the NumPy reference regardless
//! of thread count (`test_rust_backend.py::test_histogram_parity_*`). Row-wise
//! parallelism would reorder the per-cell sums and break that, so it is avoided.
//! The `binned` matrix is passed **feature-major** (`(n_features, n_rows)`, the
//! `RustSplitBackend` caches the transpose), so each feature's bins are a
//! contiguous slice and the (sorted) row gather reads them near-sequentially —
//! without that the per-feature stride across a row-major matrix is memory-bound
//! and barely scales.

use ndarray::{Array3, ArrayView1, ArrayView3};
use numpy::{
    IntoPyArray, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Minimum `rows * features` work before histogram construction goes parallel.
/// Below this rayon's dispatch overhead dominates; both branches are
/// bitwise-identical, so the threshold only trades latency, never numerics.
const PARALLEL_MIN_CELLS: usize = 1 << 17;

/// Minimum `rows * d` work before the per-leaf statistics loop goes parallel.
/// Leaf-parallelism writes disjoint per-leaf output chunks and accumulates each
/// leaf's rows in order, so it is bitwise-identical to the serial branch — the
/// threshold only trades latency, never numerics. Below it rayon's dispatch
/// overhead dominates (small trees / tiny datasets take the serial branch).
const LEAF_PARALLEL_MIN_CELLS: usize = 1 << 16;

/// Accumulate one feature's `(n_bins_max, 3)` histogram block in row order.
///
/// `hf` is the disjoint output slice for this feature (grad/hess/count
/// interleaved per bin) and `bin_row` is the feature's contiguous bins for all
/// rows. Iterating `rows` in order preserves the per-`(feature, bin)` summation
/// order shared with the NumPy backend, so the serial and parallel callers stay
/// bitwise-identical.
#[inline]
fn accumulate_feature(
    hf: &mut [f64],
    bin_row: &[u16],
    rows: &ArrayView1<'_, i64>,
    grad: &ArrayView1<'_, f64>,
    hess: &ArrayView1<'_, f64>,
) {
    for &r in rows.iter() {
        let r = r as usize;
        let b = bin_row[r] as usize;
        hf[b * 3] += grad[r];
        hf[b * 3 + 1] += hess[r];
        hf[b * 3 + 2] += 1.0;
    }
}

#[pyfunction]
fn build_histograms<'py>(
    py: Python<'py>,
    binned: PyReadonlyArray2<'py, u16>,
    rows: PyReadonlyArray1<'py, i64>,
    grad: PyReadonlyArray1<'py, f64>,
    hess: PyReadonlyArray1<'py, f64>,
    n_bins_max: usize,
) -> Bound<'py, PyArray3<f64>> {
    let binned = binned.as_array(); // feature-major: (n_features, n_rows)
    let rows = rows.as_array();
    let grad = grad.as_array();
    let hess = hess.as_array();
    let n_features = binned.shape()[0];
    let n_rows = binned.shape()[1];
    let binned_s = binned
        .as_slice()
        .expect("feature-major binned must be C-contiguous");

    let mut hist = Array3::<f64>::zeros((n_features, n_bins_max, 3));
    let h = hist.as_slice_mut().expect("freshly allocated array is contiguous");
    let chunk = n_bins_max * 3; // one feature's (n_bins_max, 3) output block

    // Feature-parallel scatter-add: output chunk f is paired with feature f's
    // contiguous bin row and accumulated in (sorted) row order. The per-feature
    // summation order matches the serial scan and NumPy, so the histogram stays
    // bitwise-identical regardless of thread count. Small nodes take the serial
    // branch to dodge rayon's dispatch overhead.
    if rows.len() * n_features < PARALLEL_MIN_CELLS {
        h.chunks_mut(chunk)
            .zip(binned_s.chunks(n_rows))
            .for_each(|(hf, bin_row)| accumulate_feature(hf, bin_row, &rows, &grad, &hess));
    } else {
        h.par_chunks_mut(chunk)
            .zip(binned_s.par_chunks(n_rows))
            .for_each(|(hf, bin_row)| accumulate_feature(hf, bin_row, &rows, &grad, &hess));
    }
    hist.into_pyarray(py)
}

#[inline]
fn leaf_score(g: f64, h: f64, l2: f64) -> f64 {
    g * g / (h + l2)
}

type Split = (i64, i64, f64, i64, i64, Option<Vec<i64>>);

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn find_best_split(
    hist: PyReadonlyArray3<'_, f64>,
    n_bins_per_feature: PyReadonlyArray1<'_, i64>,
    min_samples_leaf: i64,
    l2: f64,
    categorical_mask: PyReadonlyArray1<'_, bool>,
    cat_smooth: f64,
    min_data_per_group: i64,
    max_cat_threshold: i64,
) -> Option<Split> {
    let hist = hist.as_array();
    let n_bins = n_bins_per_feature.as_array();
    let cat_mask = categorical_mask.as_array();
    let n_features = hist.shape()[0];
    let n_bins_max = hist.shape()[1];

    // Node totals: every feature's bins partition the same rows.
    let mut g_total = 0.0;
    let mut h_total = 0.0;
    let mut n_total = 0.0;
    for b in 0..n_bins_max {
        g_total += hist[[0, b, 0]];
        h_total += hist[[0, b, 1]];
        n_total += hist[[0, b, 2]];
    }
    let parent_score = leaf_score(g_total, h_total, l2);
    let msl = min_samples_leaf as f64;

    let mut best: Option<Split> = None;
    let mut best_gain = 1e-12; // a split must strictly beat this

    // Numerical features: ordered-threshold scan, feature-major then
    // bin-minor — the same order as np.argmax over the (F, B) grid, so
    // tie-breaking matches the NumPy backend.
    for f in 0..n_features {
        if cat_mask[f] {
            continue;
        }
        let k = n_bins[f] as usize; // non-missing bins; missing bin sits at k
        if k < 2 {
            continue;
        }
        let miss_g = hist[[f, k, 0]];
        let miss_h = hist[[f, k, 1]];
        let miss_n = hist[[f, k, 2]];
        let mut cum_g = 0.0;
        let mut cum_h = 0.0;
        let mut cum_n = 0.0;
        for c in 0..(k - 1) {
            cum_g += hist[[f, c, 0]];
            cum_h += hist[[f, c, 1]];
            cum_n += hist[[f, c, 2]];
            let left_g = cum_g + miss_g;
            let left_h = cum_h + miss_h;
            let left_n = cum_n + miss_n;
            let right_n = n_total - left_n;
            if left_n < msl || right_n < msl {
                continue;
            }
            let gain = leaf_score(left_g, left_h, l2)
                + leaf_score(g_total - left_g, h_total - left_h, l2)
                - parent_score;
            if gain.is_finite() && gain > best_gain {
                best_gain = gain;
                best = Some((f as i64, c as i64, gain, left_n as i64, right_n as i64, None));
            }
        }
    }

    // Categorical features: gradient-sorted subset scan with the
    // high-cardinality guards (see the NumPy backend for the rationale).
    for f in 0..n_features {
        if !cat_mask[f] {
            continue;
        }
        if let Some(cand) = best_categorical_split(
            &hist,
            f,
            n_bins[f] as usize,
            g_total,
            h_total,
            n_total,
            parent_score,
            msl,
            l2,
            cat_smooth,
            min_data_per_group,
            max_cat_threshold,
            best_gain,
        ) {
            best_gain = cand.2;
            best = Some(cand);
        }
    }
    best
}

#[allow(clippy::too_many_arguments)]
fn best_categorical_split(
    hist: &ArrayView3<f64>,
    f: usize,
    k: usize,
    g_total: f64,
    h_total: f64,
    n_total: f64,
    parent_score: f64,
    msl: f64,
    l2: f64,
    cat_smooth: f64,
    min_data_per_group: i64,
    max_cat_threshold: i64,
    best_gain: f64,
) -> Option<Split> {
    let min_group = min_data_per_group.max(1) as f64;
    let mut present: Vec<usize> = (0..k).filter(|&b| hist[[f, b, 2]] >= min_group).collect();
    if present.len() < 2 {
        return None;
    }
    // Stable sort by smoothed Newton direction (ratios are finite since
    // hessian sums are non-negative and cat_smooth > 0).
    present.sort_by(|&a, &b| {
        let ra = hist[[f, a, 0]] / (hist[[f, a, 1]] + cat_smooth);
        let rb = hist[[f, b, 0]] / (hist[[f, b, 1]] + cat_smooth);
        ra.partial_cmp(&rb).expect("finite sort ratios")
    });

    let miss_g = hist[[f, k, 0]];
    let miss_h = hist[[f, k, 1]];
    let miss_n = hist[[f, k, 2]];
    let mut best: Option<Split> = None;
    let mut best_gain = best_gain;

    let reversed: Vec<usize> = present.iter().rev().cloned().collect();
    for order in [&present, &reversed] {
        let limit = (order.len() - 1).min(max_cat_threshold.max(0) as usize);
        let mut cum_g = 0.0;
        let mut cum_h = 0.0;
        let mut cum_n = 0.0;
        for (c, &bin) in order.iter().take(limit).enumerate() {
            cum_g += hist[[f, bin, 0]];
            cum_h += hist[[f, bin, 1]];
            cum_n += hist[[f, bin, 2]];
            let left_g = cum_g + miss_g;
            let left_h = cum_h + miss_h;
            let left_n = cum_n + miss_n;
            let right_n = n_total - left_n;
            if left_n < msl || right_n < msl {
                continue;
            }
            let gain = leaf_score(left_g, left_h, l2)
                + leaf_score(g_total - left_g, h_total - left_h, l2)
                - parent_score;
            if gain.is_finite() && gain > best_gain {
                best_gain = gain;
                let mut cats: Vec<i64> = order[..=c].iter().map(|&b| b as i64).collect();
                cats.sort_unstable();
                best = Some((
                    f as i64,
                    -1,
                    gain,
                    left_n as i64,
                    right_n as i64,
                    Some(cats),
                ));
            }
        }
    }
    best
}

/// Accumulate one leaf's fused statistics into its disjoint output chunks.
///
/// `gram_j`/`s_hz_j`/`gz_j`/`zmin_j`/`zmax_j` are this leaf's `(d*d)` / `d`
/// output slices (already zeroed / set to ±inf by the caller); the shared
/// `grad`/`hess`/`order`/`offsets`/`z_s` are read-only. Rows are visited in
/// leaf order, so the per-leaf accumulation order is identical whether this is
/// called serially or from a rayon worker — the parallel and serial branches
/// stay bitwise-identical.
#[inline]
#[allow(clippy::too_many_arguments)]
fn accumulate_leaf(
    gram_j: &mut [f64],
    s_hz_j: &mut [f64],
    gz_j: &mut [f64],
    zmin_j: &mut [f64],
    zmax_j: &mut [f64],
    li: i64,
    d: usize,
    offsets: &ArrayView1<'_, i64>,
    order: &ArrayView1<'_, i64>,
    grad: &ArrayView1<'_, f64>,
    hess: &ArrayView1<'_, f64>,
    z_s: &[f64],
) {
    let l = li as usize;
    for idx in offsets[l]..offsets[l + 1] {
        let r = order[idx as usize] as usize;
        let g = grad[r];
        let h = hess[r];
        let row = &z_s[r * d..(r + 1) * d];
        for a in 0..d {
            let za = row[a];
            s_hz_j[a] += h * za;
            gz_j[a] += g * za;
            if za < zmin_j[a] {
                zmin_j[a] = za;
            }
            if za > zmax_j[a] {
                zmax_j[a] = za;
            }
            let hza = h * za;
            let grow = a * d;
            for b in a..d {
                gram_j[grow + b] += hza * row[b];
            }
        }
    }
    // Mirror the upper triangle.
    for a in 1..d {
        for b in 0..a {
            gram_j[a * d + b] = gram_j[b * d + a];
        }
    }
}

/// Fused per-leaf statistics for embedded-linear leaf fitting (Phase 11).
///
/// One pass over the rows (in leaf order) computes, per linear-eligible
/// leaf, everything the batched normal equations need except the LAPACK
/// solve: weighted Gram matrix, gradient projection, weighted embedding
/// sums, and the extrapolation-guard min/max. The per-leaf loop is rayon
/// leaf-parallel — each leaf writes disjoint output chunks and accumulates
/// its own rows in order, so results are bitwise-identical to a serial scan
/// regardless of thread count. Parallelizing across leaves (rather than
/// threading each small per-leaf BLAS Gram, which scales poorly) is the right
/// axis, so the Python caller routes embeddings up to `_NATIVE_STATS_MAX_DIM`
/// here and only falls back to BLAS for very wide ones.
#[pyfunction]
#[allow(clippy::type_complexity)]
fn leaf_linear_stats<'py>(
    py: Python<'py>,
    z: PyReadonlyArray2<'py, f64>,
    grad: PyReadonlyArray1<'py, f64>,
    hess: PyReadonlyArray1<'py, f64>,
    order: PyReadonlyArray1<'py, i64>,
    offsets: PyReadonlyArray1<'py, i64>,
    linear: PyReadonlyArray1<'py, i64>,
) -> (
    Bound<'py, numpy::PyArray1<f64>>, // g_sum (n_leaves,)
    Bound<'py, numpy::PyArray1<f64>>, // h_sum (n_leaves,)
    Bound<'py, numpy::PyArray2<f64>>, // s_hz  (k, d)
    Bound<'py, PyArray3<f64>>,        // gram  (k, d, d)
    Bound<'py, numpy::PyArray2<f64>>, // gz    (k, d)
    Bound<'py, numpy::PyArray2<f64>>, // z_min (k, d)
    Bound<'py, numpy::PyArray2<f64>>, // z_max (k, d)
) {
    let z = z.as_array();
    let z_s = z.as_slice().expect("Z must be C-contiguous");
    let grad = grad.as_array();
    let hess = hess.as_array();
    let order = order.as_array();
    let offsets = offsets.as_array();
    let linear = linear.as_array();
    let d = z.shape()[1];
    let n_leaves = offsets.len() - 1;
    let k = linear.len();

    let mut g_sum = ndarray::Array1::<f64>::zeros(n_leaves);
    let mut h_sum = ndarray::Array1::<f64>::zeros(n_leaves);
    for l in 0..n_leaves {
        let (mut gs, mut hs) = (0.0, 0.0);
        for idx in offsets[l]..offsets[l + 1] {
            let r = order[idx as usize] as usize;
            gs += grad[r];
            hs += hess[r];
        }
        g_sum[l] = gs;
        h_sum[l] = hs;
    }

    let mut s_hz = ndarray::Array2::<f64>::zeros((k, d));
    let mut gram = Array3::<f64>::zeros((k, d, d));
    let mut gz = ndarray::Array2::<f64>::zeros((k, d));
    let mut z_min = ndarray::Array2::<f64>::from_elem((k, d), f64::INFINITY);
    let mut z_max = ndarray::Array2::<f64>::from_elem((k, d), f64::NEG_INFINITY);
    if d > 0 {
        let s_hz = s_hz.as_slice_mut().unwrap();
        let gram = gram.as_slice_mut().unwrap();
        let gz = gz.as_slice_mut().unwrap();
        let z_min = z_min.as_slice_mut().unwrap();
        let z_max = z_max.as_slice_mut().unwrap();
        let linear_s = linear
            .as_slice()
            .expect("linear indices must be C-contiguous");

        // Leaf-parallel: each leaf owns disjoint (d*d)/(d) output chunks and
        // reads the shared, immutable grad/hess/order/offsets/Z, so the rayon
        // and serial branches produce bitwise-identical per-leaf stats. Small
        // batches take the serial branch to dodge rayon's dispatch overhead
        // (same shape as build_histograms above). `order.len()` is the total
        // routed-row count (all leaves), a coarse proxy for per-leaf work.
        if k < 2 || order.len() * d < LEAF_PARALLEL_MIN_CELLS {
            gram.chunks_mut(d * d)
                .zip(s_hz.chunks_mut(d))
                .zip(gz.chunks_mut(d))
                .zip(z_min.chunks_mut(d))
                .zip(z_max.chunks_mut(d))
                .zip(linear_s.iter())
                .for_each(|(((((gram_j, s_hz_j), gz_j), zmin_j), zmax_j), &li)| {
                    accumulate_leaf(
                        gram_j, s_hz_j, gz_j, zmin_j, zmax_j, li, d, &offsets,
                        &order, &grad, &hess, z_s,
                    );
                });
        } else {
            gram.par_chunks_mut(d * d)
                .zip(s_hz.par_chunks_mut(d))
                .zip(gz.par_chunks_mut(d))
                .zip(z_min.par_chunks_mut(d))
                .zip(z_max.par_chunks_mut(d))
                .zip(linear_s.par_iter())
                .for_each(|(((((gram_j, s_hz_j), gz_j), zmin_j), zmax_j), &li)| {
                    accumulate_leaf(
                        gram_j, s_hz_j, gz_j, zmin_j, zmax_j, li, d, &offsets,
                        &order, &grad, &hess, z_s,
                    );
                });
        }
    }
    (
        g_sum.into_pyarray(py),
        h_sum.into_pyarray(py),
        s_hz.into_pyarray(py),
        gram.into_pyarray(py),
        gz.into_pyarray(py),
        z_min.into_pyarray(py),
        z_max.into_pyarray(py),
    )
}

#[pymodule]
fn repleafgbm_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_histograms, m)?)?;
    m.add_function(wrap_pyfunction!(find_best_split, m)?)?;
    m.add_function(wrap_pyfunction!(leaf_linear_stats, m)?)?;
    Ok(())
}
