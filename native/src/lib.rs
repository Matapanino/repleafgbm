//! Rust split-search kernels for RepLeafGBM.
//!
//! Implements the two functions of the `BaseSplitBackend` contract
//! (docs/backend_strategy.md, Axis 1). Semantics mirror the NumPy reference
//! kernel in `backends/numpy_backend.py`, including tie-breaking (first
//! maximum in feature-major, bin-minor order; stable category sort) and
//! floating-point accumulation order where it is observable (per-bin
//! accumulation in row order; cumulative sums with the missing block added
//! per candidate), so the two backends agree to numerical noise.

use ndarray::{Array3, ArrayView3};
use numpy::{
    IntoPyArray, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
};
use pyo3::prelude::*;

#[pyfunction]
fn build_histograms<'py>(
    py: Python<'py>,
    binned: PyReadonlyArray2<'py, u16>,
    rows: PyReadonlyArray1<'py, i64>,
    grad: PyReadonlyArray1<'py, f64>,
    hess: PyReadonlyArray1<'py, f64>,
    n_bins_max: usize,
) -> Bound<'py, PyArray3<f64>> {
    let binned = binned.as_array();
    let rows = rows.as_array();
    let grad = grad.as_array();
    let hess = hess.as_array();
    let n_features = binned.shape()[1];

    let mut hist = Array3::<f64>::zeros((n_features, n_bins_max, 3));
    let h = hist.as_slice_mut().expect("freshly allocated array is contiguous");
    for &r in rows.iter() {
        let r = r as usize;
        let g = grad[r];
        let hh = hess[r];
        for (f, &b) in binned.row(r).iter().enumerate() {
            let base = (f * n_bins_max + b as usize) * 3;
            h[base] += g;
            h[base + 1] += hh;
            h[base + 2] += 1.0;
        }
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

#[pymodule]
fn repleafgbm_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_histograms, m)?)?;
    m.add_function(wrap_pyfunction!(find_best_split, m)?)?;
    Ok(())
}
