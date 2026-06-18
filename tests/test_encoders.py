"""Tests for encoders: shapes, determinism, state round-trips."""

import numpy as np
import pytest

from repleafgbm.encoders import (
    IdentityEncoder,
    RandomProjectionEncoder,
    SimplePLREncoder,
    make_encoder,
)


@pytest.fixture
def X_num():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(100, 5))
    X[::13, 2] = np.nan
    return X


def test_identity_shape_and_standardization(X_num):
    enc = IdentityEncoder().fit(X_num)
    Z = enc.transform(X_num)
    assert Z.shape == (100, 5)
    assert enc.output_dim == 5
    assert np.isfinite(Z).all()  # NaNs imputed
    assert abs(np.nanmean(Z[:, 0])) < 1e-8
    assert abs(np.nanstd(Z[:, 0]) - 1.0) < 1e-8


def test_plr_shape_and_range(X_num):
    enc = SimplePLREncoder(n_bins=6, add_linear=False).fit(X_num)
    Z = enc.transform(X_num)
    assert Z.shape == (100, 5 * 6)
    assert enc.output_dim == 30
    assert Z.min() >= 0.0 and Z.max() <= 1.0
    # Missing values produce an all-zero block for that feature.
    nan_rows = np.isnan(X_num[:, 2])
    assert np.all(Z[nan_rows, 2 * 6 : 3 * 6] == 0.0)


def test_plr_add_linear_appends_standardized_value(X_num):
    enc = SimplePLREncoder(n_bins=4, add_linear=True).fit(X_num)
    Z = enc.transform(X_num)
    assert Z.shape == (100, 5 * 5)  # n_bins + 1 per feature
    # The linear slot of feature 0 is the standardized raw value.
    lin = Z[:, 4]
    np.testing.assert_allclose(
        lin, (X_num[:, 0] - np.nanmean(X_num[:, 0])) / np.nanstd(X_num[:, 0])
    )
    # Missing values zero the whole block, including the linear slot.
    nan_rows = np.isnan(X_num[:, 2])
    assert np.all(Z[nan_rows, 2 * 5 : 3 * 5] == 0.0)


def test_plr_linear_term_restores_extrapolation():
    """Plain PLR saturates outside the training range; the linear term must
    keep the representation moving so leaf models can extrapolate."""
    X = np.linspace(0.0, 1.0, 50).reshape(-1, 1)
    X_out = np.array([[2.0], [3.0]])  # beyond the training maximum

    plain = SimplePLREncoder(n_bins=4, add_linear=False).fit(X)
    Z_out = plain.transform(X_out)
    np.testing.assert_allclose(Z_out[0], Z_out[1])  # saturated: identical

    lin = SimplePLREncoder(n_bins=4, add_linear=True).fit(X)
    Z_out = lin.transform(X_out)
    assert Z_out[1, -1] > Z_out[0, -1]  # linear slot keeps increasing


def test_plr_monotone_in_input():
    X = np.linspace(-3, 3, 50).reshape(-1, 1)
    enc = SimplePLREncoder(n_bins=4).fit(X)
    Z = enc.transform(X)
    sums = Z.sum(axis=1)
    assert np.all(np.diff(sums) >= -1e-12)  # encoding grows with the value


def test_plr_constant_feature_does_not_crash():
    X = np.ones((30, 2))
    enc = SimplePLREncoder(n_bins=4).fit(X)
    Z = enc.transform(X)
    assert np.isfinite(Z).all()


def test_plr_huge_magnitude_constant_feature_stays_finite():
    """Regression test: '+ 1e-12' edge separation underflows at 1e15 and
    produced zero-width bins -> NaN embeddings. nextafter-based edges must
    keep the transform finite at any magnitude."""
    X = np.column_stack([np.full(40, 1e15), np.linspace(1e15, 1e15 + 1e4, 40)])
    enc = SimplePLREncoder(n_bins=4, add_linear=False).fit(X)
    Z = enc.transform(X)
    assert np.isfinite(Z).all()
    assert Z.min() >= 0.0 and Z.max() <= 1.0
    # With the linear term, values stay finite too (standardized space).
    Z_lin = SimplePLREncoder(n_bins=4, add_linear=True).fit(X).transform(X)
    assert np.isfinite(Z_lin).all()


def test_periodic_shape_determinism_and_nan(X_num):
    from repleafgbm.encoders import PeriodicEncoder

    enc = PeriodicEncoder(n_frequencies=4, random_state=7).fit(X_num)
    Z = enc.transform(X_num)
    assert Z.shape == (100, 5 * 5)  # n_frequencies + linear term
    assert enc.output_dim == 25
    assert np.isfinite(Z).all()
    # Sinusoidal slots bounded; linear slot is standardized raw value.
    sin_cols = [j * 5 + k for j in range(5) for k in range(4)]
    assert np.abs(Z[:, sin_cols]).max() <= 1.0

    # Same seed -> identical embedding (frequencies are sampled, not learned).
    Z2 = PeriodicEncoder(n_frequencies=4, random_state=7).fit(X_num).transform(X_num)
    np.testing.assert_allclose(Z, Z2)
    # Different seed -> different frequencies.
    Z3 = PeriodicEncoder(n_frequencies=4, random_state=8).fit(X_num).transform(X_num)
    assert not np.allclose(Z, Z3)


def test_periodic_state_roundtrip(X_num):
    from repleafgbm.encoders import PeriodicEncoder, make_encoder

    enc = PeriodicEncoder(n_frequencies=3, frequency_scale=2.0, random_state=1).fit(X_num)
    fresh = make_encoder(enc.name, **enc.get_config())
    fresh.set_state(enc.get_state())
    np.testing.assert_allclose(enc.transform(X_num), fresh.transform(X_num))


def test_make_encoder_injects_model_random_state():
    from repleafgbm.encoders import make_encoder

    enc = make_encoder("periodic", _default_random_state=123)
    assert enc.random_state == 123
    # Explicit encoder_params win over the injected default.
    enc = make_encoder("periodic", _default_random_state=123, random_state=5)
    assert enc.random_state == 5
    # Encoders without the argument are unaffected.
    assert make_encoder("identity", _default_random_state=123) is not None


def test_random_projection_caps_dim_and_is_deterministic(X_num):
    base = SimplePLREncoder(n_bins=8)
    enc = RandomProjectionEncoder(base, out_dim=7, random_state=42).fit(X_num)
    Z1 = enc.transform(X_num)
    assert Z1.shape == (100, 7)

    enc2 = RandomProjectionEncoder(SimplePLREncoder(n_bins=8), out_dim=7, random_state=42)
    Z2 = enc2.fit(X_num).transform(X_num)
    np.testing.assert_allclose(Z1, Z2)


def test_state_roundtrip(X_num):
    for enc in (IdentityEncoder().fit(X_num), SimplePLREncoder(n_bins=5).fit(X_num)):
        fresh = make_encoder(enc.name, **enc.get_config())
        fresh.set_state(enc.get_state())
        np.testing.assert_allclose(enc.transform(X_num), fresh.transform(X_num))


def test_unknown_encoder_name():
    with pytest.raises(ValueError, match="Unknown encoder"):
        make_encoder("nope")


# --------------------------------------------------------------------- #
# cross (Phase 16, interaction control)
# --------------------------------------------------------------------- #
def test_cross_selects_true_interaction_pair():
    from repleafgbm.encoders import CrossInteractionEncoder

    rng = np.random.default_rng(0)
    X = rng.normal(size=(1500, 6))
    y = 3.0 * X[:, 1] * X[:, 4] + rng.normal(0.0, 0.1, 1500)
    enc = CrossInteractionEncoder(n_pairs=2).fit(X, y=y)
    assert [1, 4] in enc.pairs_.tolist()
    Z = enc.transform(X)
    assert Z.shape == (1500, 6 + 2)
    assert enc.output_dim == 8
    # The selected product column explains the target far better than any
    # single raw feature does.
    prod_col = Z[:, 6 + enc.pairs_.tolist().index([1, 4])]
    assert abs(np.corrcoef(prod_col, y)[0, 1]) > 0.9


def test_cross_handles_2d_vector_target():
    """v1.4.0 multiclass/multi-output pretraining passes a ``(n, K)`` target; the
    cross encoder must score pairs across the K columns instead of crashing
    (regression test for the broadcast bug)."""
    from repleafgbm.encoders import CrossInteractionEncoder

    rng = np.random.default_rng(1)
    X = rng.normal(size=(1500, 6))
    base = 3.0 * X[:, 1] * X[:, 4]  # both outputs load on the (1, 4) product
    Y = np.column_stack([base + rng.normal(0.0, 0.1, 1500),
                         0.5 * base + rng.normal(0.0, 0.1, 1500)])
    enc = CrossInteractionEncoder(n_pairs=2).fit(X, y=Y)
    assert [1, 4] in enc.pairs_.tolist()


def test_cross_2d_single_column_matches_scalar():
    """A ``(n, 1)`` target selects the same pairs as the 1-D target (the K == 1
    reduction), so the 2-D fix leaves scalar behavior intact."""
    from repleafgbm.encoders import CrossInteractionEncoder

    rng = np.random.default_rng(3)
    X = rng.normal(size=(800, 6))
    y = 2.0 * X[:, 0] * X[:, 3] - X[:, 2] * X[:, 5] + rng.normal(0.0, 0.1, 800)
    scalar = CrossInteractionEncoder(n_pairs=3).fit(X, y=y)
    vector = CrossInteractionEncoder(n_pairs=3).fit(X, y=y[:, None])
    np.testing.assert_array_equal(scalar.pairs_, vector.pairs_)


def test_cross_without_target_and_nan(X_num):
    from repleafgbm.encoders import CrossInteractionEncoder

    enc = CrossInteractionEncoder(n_pairs=3).fit(X_num)
    Z = enc.transform(X_num)
    assert Z.shape == (100, 5 + 3)
    assert np.isfinite(Z).all()  # NaNs imputed before products


def test_cross_caps_pairs_and_roundtrips_state(X_num):
    from repleafgbm.encoders import CrossInteractionEncoder

    enc = CrossInteractionEncoder(n_pairs=100).fit(X_num, y=X_num[:, 0])
    assert enc.pairs_.shape[0] == 5 * 4 // 2  # capped at all pairs
    fresh = make_encoder("cross", **enc.get_config())
    fresh.set_state(enc.get_state())
    np.testing.assert_allclose(fresh.transform(X_num), enc.transform(X_num))


def test_cross_end_to_end(tmp_path):
    from repleafgbm import RepLeafRegressor

    rng = np.random.default_rng(2)
    X = rng.normal(size=(600, 5))
    y = X[:, 0] * X[:, 1] + 0.5 * X[:, 2] + rng.normal(0.0, 0.1, 600)
    model = RepLeafRegressor(
        n_estimators=20, num_leaves=8, min_samples_leaf=10,
        leaf_model="embedded_linear", encoder="cross",
        encoder_params={"n_pairs": 4}, random_state=42,
    )
    model.fit(X, y)
    pred = model.predict(X)
    model.save_model(tmp_path / "m")
    loaded = RepLeafRegressor.load_model(tmp_path / "m")
    np.testing.assert_allclose(loaded.predict(X), pred)
