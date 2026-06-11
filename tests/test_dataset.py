"""Tests for RepLeafDataset and feature preprocessing."""

import numpy as np
import pandas as pd
import pytest

from repleafgbm.data import RepLeafDataset


def _make_df():
    return pd.DataFrame(
        {
            "num1": [1.0, 2.0, 3.0, 4.0],
            "num2": [0.5, np.nan, 1.5, 2.0],
            "cat1": ["a", "b", "a", None],
        }
    )


def test_dataframe_with_categoricals():
    df = _make_df()
    ds = RepLeafDataset(df, y=[1.0, 2.0, 3.0, 4.0], categorical_features=["cat1"])
    assert ds.n_rows == 4
    assert ds.n_features == 3
    assert ds.metadata.categorical_features == ["cat1"]
    assert ds.metadata.numerical_features == ["num1", "num2"]

    X_raw = ds.get_raw_features()
    assert X_raw.dtype == np.float64
    # "a" -> 0, "b" -> 1 (sorted), missing -> NaN
    cat_col = X_raw[:, ds.feature_names.index("cat1")]
    assert cat_col[0] == 0.0 and cat_col[1] == 1.0 and cat_col[2] == 0.0
    assert np.isnan(cat_col[3])


def test_categorical_autodetection():
    df = _make_df()
    ds = RepLeafDataset(df)
    assert ds.metadata.categorical_features == ["cat1"]


def test_unseen_category_becomes_nan():
    df = _make_df()
    ds = RepLeafDataset(df, categorical_features=["cat1"])
    df_new = _make_df()
    df_new.loc[0, "cat1"] = "unseen"
    ds_new = RepLeafDataset(df_new, metadata=ds.metadata)
    cat_col = ds_new.get_raw_features()[:, ds.feature_names.index("cat1")]
    assert np.isnan(cat_col[0])


def test_numpy_input():
    X = np.arange(12, dtype=float).reshape(4, 3)
    ds = RepLeafDataset(X, y=np.arange(4))
    assert ds.feature_names == ["f0", "f1", "f2"]
    assert ds.metadata.categorical_features == []
    np.testing.assert_array_equal(ds.get_raw_features(), X)


def test_numerical_view_excludes_categoricals():
    df = _make_df()
    ds = RepLeafDataset(df, categorical_features=["cat1"])
    assert ds.get_numerical_features().shape == (4, 2)


def test_y_length_mismatch_raises():
    with pytest.raises(ValueError, match="rows"):
        RepLeafDataset(np.zeros((4, 2)), y=[1.0, 2.0])


def test_embedding_cache():
    from repleafgbm.encoders import make_encoder

    X = np.random.default_rng(0).normal(size=(20, 3))
    ds = RepLeafDataset(X)
    enc = make_encoder("identity").fit(ds.get_numerical_features())
    Z1 = ds.get_embeddings(enc)
    Z2 = ds.get_embeddings(enc)
    assert Z1 is Z2  # cached


def test_embedding_cache_invalidated_on_encoder_switch():
    from repleafgbm.encoders import make_encoder

    X = np.random.default_rng(0).normal(size=(20, 3))
    ds = RepLeafDataset(X)
    enc_id = make_encoder("identity").fit(ds.get_numerical_features())
    enc_plr = make_encoder("plr", n_bins=4).fit(ds.get_numerical_features())

    Z_id = ds.get_embeddings(enc_id)
    Z_plr = ds.get_embeddings(enc_plr)  # different encoder -> recompute
    assert Z_plr.shape == (20, 3 * (4 + 1))  # n_bins + linear term per feature
    # Switching back recomputes correctly instead of serving a stale entry.
    np.testing.assert_allclose(ds.get_embeddings(enc_id), Z_id)
