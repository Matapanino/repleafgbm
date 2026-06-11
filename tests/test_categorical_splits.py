"""Tests for native categorical subset splits (Phase 8)."""

import json

import numpy as np
import pandas as pd

from repleafgbm import RepLeafClassifier, RepLeafDataset, RepLeafRegressor


def _subset_frame(n=600, seed=0, noise=0.05):
    """Target depends on a NON-contiguous category subset: codes for
    {a,d,e} are {0,3,4} after sorted-ordinal encoding, so a single ordered
    threshold cannot separate the groups but one subset split can."""
    rng = np.random.default_rng(seed)
    cat = rng.choice(list("abcde"), size=n)
    high = np.isin(cat, ["a", "d", "e"])
    y = np.where(high, 5.0, -5.0) + rng.normal(0.0, noise, n)
    df = pd.DataFrame({"c": cat, "x": rng.normal(size=n)})
    return df, y


def test_single_subset_split_separates_noncontiguous_groups():
    df, y = _subset_frame()
    model = RepLeafRegressor(
        n_estimators=1, learning_rate=1.0, num_leaves=2,  # exactly one split
        min_samples_leaf=10, leaf_model="constant", random_state=42,
    )
    model.fit(RepLeafDataset(df, y, categorical_features=["c"]))

    tree = model.booster_.trees_[0]
    assert tree.left_categories is not None
    cats = [c for c in tree.left_categories if c is not None]
    assert len(cats) == 1  # the single split is categorical
    # The chosen subset is exactly one of the two true groups.
    assert set(cats[0].astype(int)) in ({0, 3, 4}, {1, 2})

    pred = model.predict(df)
    assert float(np.sqrt(np.mean((pred - y) ** 2))) < 0.2  # group means recovered


def test_ordered_threshold_cannot_do_this_in_one_split():
    """Control: same data with the category treated as numeric codes needs
    more than one split, so a 2-leaf tree must do much worse."""
    df, y = _subset_frame()
    codes = df["c"].map({c: i for i, c in enumerate(sorted(df["c"].unique()))})
    X = np.column_stack([codes.to_numpy(float), df["x"].to_numpy()])
    model = RepLeafRegressor(
        n_estimators=1, learning_rate=1.0, num_leaves=2,
        min_samples_leaf=10, leaf_model="constant", random_state=42,
    )
    model.fit(X, y)  # no categorical declaration -> ordered scan
    pred = model.predict(X)
    assert float(np.sqrt(np.mean((pred - y) ** 2))) > 2.0


def test_apply_matches_training_partition_and_routing_rules():
    df, y = _subset_frame(n=800, seed=3, noise=0.3)
    train = RepLeafDataset(df, y, categorical_features=["c"])
    model = RepLeafRegressor(
        n_estimators=60, num_leaves=8, min_samples_leaf=10,
        leaf_model="embedded_linear", random_state=42,
    )
    model.fit(train)

    # Missing and unseen categories route via NaN -> missing_left (left).
    df_new = df.iloc[:6].copy()
    df_new.loc[df_new.index[0], "c"] = None
    df_new.loc[df_new.index[1], "c"] = "unseen"
    pred = model.predict(df_new)
    assert np.isfinite(pred).all()

    # In-sample predictions are consistent between fit-time partitions and
    # apply()-based prediction (training loss is small).
    pred_tr = model.predict(df)
    assert float(np.sqrt(np.mean((pred_tr - y) ** 2))) < 1.0


def test_categorical_split_roundtrip(tmp_path):
    df, y = _subset_frame(n=500, seed=5, noise=0.2)
    train = RepLeafDataset(df, y, categorical_features=["c"])
    model = RepLeafRegressor(
        n_estimators=8, num_leaves=6, min_samples_leaf=10, random_state=42
    )
    model.fit(train)
    assert any(t.left_categories is not None for t in model.booster_.trees_)
    pred = model.predict(df)

    model.save_model(tmp_path / "m")
    cfg = json.loads((tmp_path / "m" / "model_config.json").read_text())
    assert cfg["format_version"] == 3
    loaded = RepLeafRegressor.load_model(tmp_path / "m")
    np.testing.assert_allclose(loaded.predict(df), pred)


def test_min_samples_leaf_respected_with_categorical_splits():
    df, y = _subset_frame(n=600, seed=7, noise=0.5)
    train = RepLeafDataset(df, y, categorical_features=["c"])
    model = RepLeafRegressor(
        n_estimators=3, num_leaves=16, min_samples_leaf=40,
        leaf_model="constant", random_state=42,
    )
    model.fit(train)
    X_raw = train.get_raw_features()
    for tree in model.booster_.trees_:
        counts = np.bincount(tree.apply(X_raw))
        assert counts[counts > 0].min() >= 40


def test_determinism_with_categorical_splits():
    df, y = _subset_frame(n=500, seed=9, noise=0.3)
    kwargs = dict(n_estimators=8, num_leaves=8, min_samples_leaf=10, random_state=42)
    p1 = (RepLeafRegressor(**kwargs)
          .fit(RepLeafDataset(df, y, categorical_features=["c"])).predict(df))
    p2 = (RepLeafRegressor(**kwargs)
          .fit(RepLeafDataset(df, y, categorical_features=["c"])).predict(df))
    np.testing.assert_allclose(p1, p2)


def test_high_cardinality_falls_back_to_ordered():
    rng = np.random.default_rng(11)
    n = 400
    cat = rng.integers(0, 300, n).astype(str)  # 300 categories
    y = rng.normal(size=n)
    df = pd.DataFrame({"c": cat, "x": rng.normal(size=n)})
    model = RepLeafRegressor(
        n_estimators=3, num_leaves=4, min_samples_leaf=10,
        max_bins=128, random_state=42,  # 300 > 128 -> ordered fallback
    )
    model.fit(RepLeafDataset(df, y, categorical_features=["c"]))
    assert all(t.left_categories is None for t in model.booster_.trees_)
    assert model.predict(df).shape == (n,)


def test_classifier_with_categorical_splits():
    rng = np.random.default_rng(13)
    n = 800
    cat = rng.choice(list("abcde"), size=n)
    logit = np.where(np.isin(cat, ["b", "e"]), 2.0, -2.0) + rng.normal(0, 0.5, n)
    y = (logit > 0).astype(int)
    df = pd.DataFrame({"c": cat, "x": rng.normal(size=n)})
    model = RepLeafClassifier(
        n_estimators=20, num_leaves=4, min_samples_leaf=10, random_state=42
    )
    model.fit(RepLeafDataset(df, y, categorical_features=["c"]))
    acc = (model.predict(df) == y).mean()
    assert acc > 0.9
