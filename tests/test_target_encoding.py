"""Fold-safe target-encoding representation tests."""

from __future__ import annotations

import numpy as np

from repleafgbm import RepLeafClassifier, RepLeafDataset, RepLeafRegressor
from repleafgbm.encoders import (
    TargetEncodingEncoder,
    encoder_from_config,
)


def test_shape_column_order_and_weighted_values_are_finite():
    X = np.array([[0.0, 10.0], [0.0, 20.0], [1.0, 10.0], [1.0, 20.0]])
    y = np.array([1.0, 3.0, 5.0, 9.0])
    weight = np.array([1.0, 3.0, 1.0, 1.0])
    encoder = TargetEncodingEncoder(
        columns=[1, 0], scales=["exact"], alpha=0.0, add_count=True
    ).fit(X, y, weight)

    transformed = encoder.transform(np.array([[0.0, 10.0], [1.0, 20.0]]))

    assert transformed.shape == (2, 4)
    assert np.isfinite(transformed).all()
    # (column 1 mean, column 1 count, column 0 mean, column 0 count)
    np.testing.assert_allclose(
        transformed,
        [[3.0, np.log1p(2.0), 2.5, np.log1p(4.0)], [4.5, np.log1p(4.0), 7.0, np.log1p(2.0)]],
    )


def test_own_target_does_not_change_own_oof_value_but_changes_full_transform():
    X = np.repeat(np.arange(6, dtype=np.float64), 4).reshape(-1, 1)
    y = np.linspace(-2.0, 3.0, len(X))
    changed = y.copy()
    changed[0] += 1000.0
    params = dict(n_folds=4, random_state=17, alpha=5.0, add_count=False)

    first = TargetEncodingEncoder(**params)
    second = TargetEncodingEncoder(**params)
    first_oof = first.fit_transform_train(X, y)
    second_oof = second.fit_transform_train(X, changed)

    np.testing.assert_array_equal(first_oof[0], second_oof[0])
    assert not np.array_equal(first.transform(X[[0]]), second.transform(X[[0]]))


def test_same_seed_is_deterministic_and_state_round_trips():
    rng = np.random.default_rng(29)
    X = rng.normal(size=(80, 2))
    X[::9, 1] = np.nan
    y = rng.normal(size=80)
    params = dict(
        columns=[0, 1],
        scales=["exact", 1, "bin:4"],
        alpha=5.0,
        n_folds=5,
        add_count=True,
        backoff=True,
        min_count=3,
        random_state=11,
    )
    first = TargetEncodingEncoder(**params)
    second = TargetEncodingEncoder(**params)

    np.testing.assert_array_equal(first.fit_transform_train(X, y), second.fit_transform_train(X, y))
    state = first.get_state()
    for column in (0, 1):
        for scale in ("exact", 1, "bin:4"):
            for statistic in ("keys", "sums", "counts"):
                assert f"{column}_{scale}_{statistic}" in state
    restored = encoder_from_config(first.name, first.get_config())
    restored.set_state(state)
    probe = np.vstack([X[:8], [[100.0, np.nan]]])
    np.testing.assert_array_equal(first.transform(probe), restored.transform(probe))


def test_unknown_key_uses_prior_and_nan_is_a_learned_key():
    X = np.array([[0.0], [1.0], [np.nan], [np.nan]])
    y = np.array([0.0, 2.0, 8.0, 10.0])
    encoder = TargetEncodingEncoder(alpha=0.0, add_count=True).fit(X, y)

    transformed = encoder.transform(np.array([[99.0], [np.nan]]))

    np.testing.assert_allclose(transformed[0], [5.0, 0.0])
    np.testing.assert_allclose(transformed[1], [9.0, np.log1p(2.0)])


def test_sparse_weights_use_full_prior_when_a_fold_complement_has_no_mass():
    X = np.arange(4, dtype=np.float64).reshape(-1, 1)
    y = np.array([7.0, 20.0, 30.0, 40.0])
    weight = np.array([1.0, 0.0, 0.0, 0.0])
    encoder = TargetEncodingEncoder(
        n_folds=2, random_state=42, alpha=0.0, add_count=False
    )

    transformed = encoder.fit_transform_train(X, y, weight)

    np.testing.assert_array_equal(transformed, np.full((4, 1), 7.0))


def test_exact_round_and_bin_scales_have_the_documented_width():
    X = np.arange(24, dtype=np.float64).reshape(8, 3)
    y = np.arange(8, dtype=np.float64)
    encoder = TargetEncodingEncoder(
        columns=[0, 2], scales=["exact", 0, "bin:4"], add_count=True
    ).fit(X, y)

    assert encoder.output_dim == 2 * 3 * 2
    assert encoder.transform(X).shape == (8, 12)


def test_backoff_replaces_rare_exact_mean_with_next_coarser_scale():
    X = np.array([[0.11], [0.12], [1.11], [1.12]])
    y = np.array([0.0, 2.0, 8.0, 10.0])
    encoder = TargetEncodingEncoder(
        scales=["exact", 0],
        alpha=0.0,
        add_count=False,
        backoff=True,
        min_count=2,
    ).fit(X, y)

    transformed = encoder.transform(X)

    np.testing.assert_allclose(transformed[:, 0], [1.0, 1.0, 9.0, 9.0])
    np.testing.assert_allclose(transformed[:, 1], [1.0, 1.0, 9.0, 9.0])


def test_classifier_target_encoding_expert_fit_predict_save_load(tmp_path):
    rng = np.random.default_rng(37)
    n = 320
    route = rng.normal(size=n)
    category_code = rng.integers(0, 12, size=n).astype(np.float64)
    signal = (category_code % 3) - 0.3 * route + rng.normal(0.0, 0.25, n)
    X = np.column_stack([route, category_code])
    y = (signal > np.median(signal)).astype(int)
    model = RepLeafClassifier(
        n_estimators=5,
        num_leaves=5,
        min_samples_leaf=12,
        leaf_model="select",
        leaf_experts=[
            {
                "name": "te",
                "encoder": "target_encoding",
                "columns": [1],
                "encoder_params": {
                    "scales": ["exact", 0, "bin:4"],
                    "n_folds": 4,
                    "random_state": 7,
                },
            }
        ],
        leaf_reduction="none",
        split_backend="numpy",
        random_state=7,
    ).fit(X, y)

    before = model.predict_proba(X[:20])
    assert before.shape == (20, 2)
    assert "te" in model.expert_selection_counts_
    path = tmp_path / "te-model"
    model.save_model(path)
    restored = RepLeafClassifier.load_model(path)
    np.testing.assert_array_equal(before, restored.predict_proba(X[:20]))


def test_estimator_bundle_training_cache_is_not_reused_for_prediction():
    rng = np.random.default_rng(41)
    X = np.column_stack([np.repeat(np.arange(10), 6), rng.normal(size=60)]).astype(float)
    y = 0.5 * X[:, 0] + rng.normal(size=60)
    X_eval = np.column_stack([np.arange(12) % 10, rng.normal(size=12)]).astype(float)
    y_eval = 0.5 * X_eval[:, 0]
    train = RepLeafDataset(X, y)
    valid = RepLeafDataset(X_eval, y_eval, metadata=train.metadata)
    encoder_params = dict(alpha=0.0, n_folds=3, add_count=False, random_state=13)
    model = RepLeafRegressor(
        n_estimators=2,
        num_leaves=3,
        min_samples_leaf=5,
        leaf_model="embedded_linear",
        leaf_experts=[
            {
                "name": "te",
                "encoder": "target_encoding",
                "columns": [0],
                "encoder_params": encoder_params,
            }
        ],
        leaf_reduction="none",
        split_backend="numpy",
        random_state=13,
    ).fit(train, eval_set=[valid])

    np.testing.assert_array_equal(model.predict(train), model.predict(X))
    np.testing.assert_array_equal(
        valid.get_embeddings(model.encoder_), model.encoder_.transform(X_eval)
    )


def test_direct_encoder_training_cache_is_not_reused_for_prediction():
    X = np.repeat(np.arange(8, dtype=np.float64), 6).reshape(-1, 1)
    y = np.linspace(-3.0, 4.0, len(X))
    train = RepLeafDataset(X, y)
    encoder_params = dict(alpha=0.0, n_folds=3, add_count=False, random_state=19)
    model = RepLeafRegressor(
        n_estimators=1,
        num_leaves=2,
        min_samples_leaf=5,
        leaf_model="embedded_linear",
        encoder="target_encoding",
        encoder_params=encoder_params,
        leaf_reduction="none",
        split_backend="numpy",
        random_state=19,
    ).fit(train)

    np.testing.assert_array_equal(model.predict(train), model.predict(X))


def test_multiclass_target_is_rejected():
    X = np.arange(12, dtype=np.float64).reshape(6, 2)
    y = np.eye(3, dtype=np.float64)[[0, 1, 2, 0, 1, 2]]

    with np.testing.assert_raises_regex(NotImplementedError, "scalar target only in v1"):
        TargetEncodingEncoder().fit(X, y)
