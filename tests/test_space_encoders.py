"""CPU space-bank encoders: contracts, numerical behavior, and model persistence."""

import inspect
import io
import json
from statistics import NormalDist

import numpy as np
import pytest

from repleafgbm import RepLeafRegressor
from repleafgbm.encoders import (
    PCAEncoder,
    PrototypeDistanceEncoder,
    QuantileEncoder,
    RandomFourierEncoder,
    RowStatsEncoder,
    SplineEncoder,
    encoder_from_config,
    make_encoder,
)

CASES = [
    ("pca", {}, 5),
    ("pca", {"n_components": 3, "whiten": True}, 3),
    ("quantile", {}, 5),
    ("quantile", {"n_quantiles": 17, "output": "normal", "add_tail": True}, 15),
    ("spline", {}, 25),
    ("spline", {"n_knots": 2}, 15),
    ("rff", {}, 32),
    ("rff", {"n_components": 7, "gamma": 0.3}, 14),
    ("row_stats", {}, 8),
    ("row_stats", {"columns": [4, 1, 0]}, 8),
    ("prototype", {}, 8),
    ("prototype", {"n_prototypes": 3, "max_iter": 7, "transform": "rbf"}, 3),
]
NAMES = ("pca", "quantile", "spline", "rff", "row_stats", "prototype")


@pytest.fixture
def space_data():
    X = np.random.default_rng(71).normal(size=(240, 5))
    X[::11, 2] = np.nan
    X[0] = np.nan
    return X


@pytest.mark.parametrize("name,params,dim", CASES)
def test_space_contract_and_exact_roundtrip(name, params, dim, space_data):
    enc = make_encoder(name, _default_random_state=19, **params).fit(space_data)
    probe = np.vstack([space_data[:31], np.full((1, 5), 20.0)])
    Z = enc.transform(probe)
    assert Z.shape == (32, dim)
    assert enc.output_dim == dim
    assert Z.dtype == np.float64
    assert np.isfinite(Z).all()
    assert enc.transform(np.empty((0, 5))).shape == (0, dim)
    again = make_encoder(name, _default_random_state=19, **params).fit(space_data)
    assert np.array_equal(Z, again.transform(probe))
    config = json.loads(json.dumps(enc.get_config()))
    assert set(config) == set(inspect.signature(type(enc)).parameters)
    state = enc.get_state()
    assert all(isinstance(value, np.ndarray) for value in state.values())
    buffer = io.BytesIO()
    np.savez(buffer, **state)
    buffer.seek(0)
    restored = encoder_from_config(name, config)
    with np.load(buffer, allow_pickle=False) as saved:
        restored.set_state(dict(saved))
    assert restored.output_dim == dim
    assert np.array_equal(Z, restored.transform(probe))


@pytest.mark.parametrize("name,params,dim", CASES)
@pytest.mark.parametrize("value", [0.0, np.nan])
def test_constant_and_all_missing_columns(name, params, dim, value):
    X = np.full((20, 5), value)
    enc = make_encoder(name, **params).fit(X)
    assert np.isfinite(enc.transform(X)).all()


@pytest.mark.parametrize("name", NAMES)
def test_space_input_validation(name, space_data):
    enc = make_encoder(name)
    for X, message in [(np.empty((10, 0)), "column"), (np.empty((0, 5)), "row"),
                       (np.ones(5), "2-D"), (np.full((20, 5), np.inf), "finite")]:
        with pytest.raises(ValueError, match=message):
            enc.fit(X)
    enc.fit(space_data)
    with pytest.raises(ValueError, match="2-D|features|columns"):
        enc.transform(np.ones(5))
    if name != "row_stats":  # Row aggregates can be applied to any nonempty width.
        with pytest.raises(ValueError, match="features"):
            enc.transform(np.ones((3, 6)))


@pytest.mark.parametrize("name", NAMES)
def test_fitted_guard(name):
    enc = make_encoder(name)
    with pytest.raises(RuntimeError, match="fitted"):
        enc.transform(np.ones((3, 5)))
    with pytest.raises(RuntimeError, match="fitted"):
        enc.get_state()
    with pytest.raises(RuntimeError, match="fitted"):
        _ = enc.output_dim


@pytest.mark.parametrize("name,params,message", [
    ("pca", {"n_components": 0}, "n_components"),
    ("pca", {"n_components": 1.5}, "n_components"),
    ("quantile", {"n_quantiles": 1}, "n_quantiles"),
    ("quantile", {"output": "invalid"}, "output"),
    ("spline", {"n_knots": 0}, "n_knots"),
    ("rff", {"n_components": 0}, "n_components"),
    ("rff", {"gamma": 0.0}, "gamma"),
    ("rff", {"gamma": float("nan")}, "gamma"),
    ("rff", {"gamma": "auto"}, "gamma"),
    ("prototype", {"n_prototypes": 0}, "n_prototypes"),
    ("prototype", {"max_iter": 0}, "max_iter"),
    ("prototype", {"transform": "invalid"}, "transform"),
    ("row_stats", {"columns": []}, "columns"),
    ("row_stats", {"columns": [0, 0]}, "columns"),
    ("row_stats", {"columns": [-1]}, "columns"),
    ("row_stats", {"columns": [1.5]}, "columns"),
])
def test_invalid_constructor_arguments(name, params, message):
    with pytest.raises(ValueError, match=message):
        make_encoder(name, **params)


@pytest.mark.parametrize("name,params,rows", [
    ("pca", {}, 4), ("pca", {"n_components": 1, "whiten": True}, 1),
    ("rff", {}, 1), ("prototype", {}, 7),
])
def test_insufficient_training_rows(name, params, rows):
    with pytest.raises(ValueError, match="row"):
        make_encoder(name, **params).fit(np.ones((rows, 5)))


@pytest.mark.parametrize("name", ["rff", "prototype"])
def test_seed_injection_and_override(name, space_data):
    injected = make_encoder(name, _default_random_state=123).fit(space_data)
    explicit = make_encoder(name, _default_random_state=99, random_state=123).fit(space_data)
    changed = make_encoder(name, _default_random_state=124).fit(space_data)
    assert injected.random_state == explicit.random_state == 123
    assert np.array_equal(injected.transform(space_data), explicit.transform(space_data))
    assert not np.array_equal(injected.transform(space_data), changed.transform(space_data))


def test_registry_exports_and_error_candidates():
    classes = (PCAEncoder, QuantileEncoder, SplineEncoder, RandomFourierEncoder,
               RowStatsEncoder, PrototypeDistanceEncoder)
    for name, cls in zip(NAMES, classes):
        assert isinstance(make_encoder(name), cls)
    with pytest.raises(ValueError) as error:
        make_encoder("missing_space")
    for name in NAMES:
        assert repr(name) in str(error.value)


def test_pca_orthogonality_whitening_and_clipping():
    X = np.random.default_rng(2).normal(size=(80, 5))
    enc = PCAEncoder(n_components=99, whiten=True).fit(X)
    assert enc.output_dim == 5
    np.testing.assert_allclose(enc.components_ @ enc.components_.T, np.eye(5), atol=1e-12)
    np.testing.assert_allclose(np.cov(enc.transform(X), rowvar=False), np.eye(5), atol=1e-12)
    plain = PCAEncoder(n_components=3).fit(X)
    standardized = (X - X.mean(axis=0)) / X.std(axis=0)
    _, singular_values, _ = np.linalg.svd(standardized, full_matrices=False)
    np.testing.assert_allclose(plain.singular_values_, singular_values[:3])


def test_quantile_rank_ties_tail_and_normal_values():
    X = np.arange(5.0)[:, None]
    enc = QuantileEncoder(n_quantiles=5, add_tail=True).fit(X)
    probe = np.array([[-2.0], [0.0], [1.0], [2.0], [4.0], [7.0], [np.nan]])
    Z = enc.transform(probe)
    np.testing.assert_allclose(Z[:, 0], [0, 0, 0.25, 0.5, 1, 1, 0])
    np.testing.assert_allclose(Z[:, 1:], [[2, 0], [0, 0], [0, 0], [0, 0],
                                       [0, 0], [0, 3], [0, 0]])
    assert ((Z[:, 0] >= 0) & (Z[:, 0] <= 1)).all()
    tied = QuantileEncoder(n_quantiles=5).fit(np.array([[0], [1], [1], [1], [2]]))
    assert tied.transform(np.array([[1]]))[0, 0] == 0.5
    normal = QuantileEncoder(n_quantiles=5, output="normal").fit(X).transform(X)
    expected = [NormalDist().inv_cdf(p) for p in (0.25, 0.5, 0.75)]
    np.testing.assert_allclose(normal[1:4, 0], expected, atol=1e-8)
    assert np.isfinite(normal).all()
    assert np.all(np.diff(normal[:, 0]) > 0)


def test_quantile_normal_tail_accuracy():
    # A two-point table maps [0, 1] to probabilities exactly, testing both
    # rational-approximation branches against an independent stdlib oracle.
    probabilities = np.unique(np.concatenate([
        np.geomspace(1e-7, 0.5, 100), 1 - np.geomspace(1e-7, 0.5, 100),
        [0.02425, 1 - 0.02425],
    ]))
    enc = QuantileEncoder(n_quantiles=2, output="normal").fit(np.array([[0.0], [1.0]]))
    expected = np.array([NormalDist().inv_cdf(p) for p in probabilities])
    np.testing.assert_allclose(enc.transform(probabilities[:, None])[:, 0],
                               expected, rtol=0, atol=1e-8)


@pytest.mark.parametrize("name", NAMES)
def test_fixed_spaces_ignore_targets_and_weights(name, space_data):
    plain = make_encoder(name).fit(space_data)
    supervised = make_encoder(name).fit(
        space_data, y=np.ones((len(space_data), 2)),
        sample_weight=np.linspace(0.5, 2.0, len(space_data)),
    )
    assert np.array_equal(plain.transform(space_data), supervised.transform(space_data))


def test_spline_hinges_linear_extrapolation_and_missing_block():
    X = np.linspace(0, 1, 30)[:, None]
    enc = SplineEncoder(n_knots=3).fit(X)
    probe = np.array([[-3.0], [-2.0], [2.0], [3.0], [np.nan]])
    Z = enc.transform(probe)
    linear = (probe[:4, 0] - X.mean()) / X.std()
    np.testing.assert_allclose(Z[:4, -1], linear)
    np.testing.assert_allclose(Z[:4, :-1], np.maximum(linear[:, None] - enc.knots_[0], 0))
    assert Z[1, -1] > Z[0, -1] and Z[3, -1] > Z[2, -1]
    assert (Z[-1] == 0).all()


def test_row_stats_hand_calculation_and_selected_order():
    enc = RowStatsEncoder()
    X = np.array([[1, 3, 5, 7], [0, np.nan, 4, 6], [np.nan] * 4])
    Z = enc.fit(X).transform(X)
    raw = np.array([
        [4, np.sqrt(5), 1, 7, 0, 0, 2, -4],
        [10 / 3, np.std([0, 4, 6]), 0, 6, 1, 1, 2, -5],
        [0, 0, 0, 0, 0, 4, 0, 0],
    ])
    expected_scale = raw.std(axis=0)
    expected_scale[expected_scale == 0] = 1.0
    np.testing.assert_allclose(enc.mean_, raw.mean(axis=0))
    np.testing.assert_allclose(enc.scale_, expected_scale)
    np.testing.assert_allclose(Z, (raw - raw.mean(axis=0)) / expected_scale)
    # Missing values are excluded; slope uses observed original positions (0, 2, 3).
    selected_encoder = RowStatsEncoder(columns=[3, 1]).fit(X)
    selected_raw = np.array([
        [5, 2, 3, 7, 0, 0, -4, 4],
        [6, 0, 6, 6, 0, 1, 0, 6],
        [0, 0, 0, 0, 0, 2, 0, 0],
    ])
    selected_scale = selected_raw.std(axis=0)
    selected_scale[selected_scale == 0] = 1.0
    np.testing.assert_allclose(
        selected_encoder.transform(X[:1]),
        ((selected_raw - selected_raw.mean(axis=0)) / selected_scale)[:1],
    )
    assert set(enc.get_state()) == {"mean", "scale"}
    with pytest.raises(ValueError, match="columns|index"):
        RowStatsEncoder(columns=[4]).fit(X)


def test_row_stats_regressor_is_invariant_to_feature_position_shift():
    rng = np.random.default_rng(87)
    X = rng.normal(size=(240, 5))
    y = 1.3 * X[:, 0] - 0.8 * X[:, 2] + np.sin(X[:, 4])
    params = dict(
        encoder="row_stats", leaf_model="embedded_linear", n_estimators=8,
        num_leaves=7, min_samples_leaf=12, leaf_reduction="none", random_state=31,
    )
    pred = RepLeafRegressor(**params).fit(X, y).predict(X)
    shifted = RepLeafRegressor(**params).fit(X + 1e7, y).predict(X + 1e7)
    np.testing.assert_allclose(pred, shifted, rtol=1e-9, atol=1e-9)


def test_rff_kernel_normalization_and_median_bandwidth():
    X = np.arange(4.0)[:, None]
    enc = RandomFourierEncoder(n_components=9).fit(X)
    Z = enc.transform(X)
    np.testing.assert_allclose((Z * Z).sum(axis=1), 1.0, atol=1e-12)
    standardized = (X[:, 0] - X.mean()) / X.std()
    distances = [abs(a - b) for i, a in enumerate(standardized) for b in standardized[i + 1:]]
    assert enc.gamma_ == pytest.approx(1 / (2 * np.median(distances) ** 2))
    large = np.random.default_rng(9).normal(size=(2050, 2))
    first = RandomFourierEncoder(random_state=14).fit(large)
    second = RandomFourierEncoder(random_state=14).fit(large)
    assert np.array_equal(first.transform(large[:10]), second.transform(large[:10]))


def test_prototype_distances_lloyd_and_rbf_bandwidth():
    X = np.array([[-4.0], [-3.0], [3.0], [4.0]])
    enc = PrototypeDistanceEncoder(n_prototypes=2, random_state=3).fit(X)
    expected = np.array([-3.5, 3.5]) / X.std()
    np.testing.assert_allclose(np.sort(enc.centers_[:, 0]), expected)
    distances = enc.transform(X)
    assert (distances >= 0).all()
    assert enc.sigma_ == pytest.approx(expected[1] - expected[0])
    rbf = PrototypeDistanceEncoder(n_prototypes=2, transform="rbf", random_state=3).fit(X)
    np.testing.assert_allclose(rbf.transform(X), np.exp(-distances**2 / (2 * enc.sigma_**2)))
    single = PrototypeDistanceEncoder(n_prototypes=1).fit(X)
    np.testing.assert_allclose(single.centers_, [[0.0]], atol=1e-12)
    assert single.sigma_ == 1.0


@pytest.mark.parametrize("name,params,dim", CASES)
def test_space_regressor_fit_predict_save_load(name, params, dim, space_data, tmp_path):
    X = space_data
    y = np.nan_to_num(X[:, 0]) + 0.3 * np.nan_to_num(X[:, 1]) ** 2
    model = RepLeafRegressor(
        encoder=name, encoder_params=params, leaf_model="embedded_linear",
        n_estimators=3, num_leaves=3, min_samples_leaf=30,
        leaf_reduction="none", random_state=23,
    ).fit(X, y)
    pred = model.predict(X[:40])
    assert pred.shape == (40,)
    assert np.isfinite(pred).all()
    assert model.leaf_embedding_dim_ == dim
    assert model.linear_leaf_fraction_ > 0
    model.save_model(tmp_path / "model")
    restored = RepLeafRegressor.load_model(tmp_path / "model")
    assert np.array_equal(pred, restored.predict(X[:40]))
