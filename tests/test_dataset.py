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


def test_pandas_categorical_declared_order_preserved():
    df = _make_df()
    # Declared order "b" < "a", plus "c" declared but never observed.
    df["cat1"] = pd.Categorical(df["cat1"], categories=["b", "a", "c"])
    ds = RepLeafDataset(df)
    assert ds.metadata.category_maps["cat1"] == ["b", "a", "c"]
    cat_col = ds.get_raw_features()[:, ds.feature_names.index("cat1")]
    # "a" -> 1, "b" -> 0 (declared order, not sorted), missing -> NaN
    assert cat_col[0] == 1.0 and cat_col[1] == 0.0 and cat_col[2] == 1.0
    assert np.isnan(cat_col[3])

    # The declared-but-unobserved category has a stable code at predict time.
    df_new = _make_df()
    df_new.loc[0, "cat1"] = "c"
    ds_new = RepLeafDataset(df_new, metadata=ds.metadata)
    assert ds_new.get_raw_features()[0, ds.feature_names.index("cat1")] == 2.0


def test_frequency_encoding():
    df = pd.DataFrame(
        {
            "num1": [1.0, 2.0, 3.0, 4.0],
            "city": ["tokyo", "tokyo", "osaka", None],
        }
    )
    ds = RepLeafDataset(df, frequency_encoded_features=["city"])
    # Frequency-encoded columns are numerical downstream.
    assert ds.metadata.categorical_features == []
    assert "city" in ds.metadata.numerical_features
    assert ds.get_numerical_features().shape == (4, 2)

    col = ds.get_raw_features()[:, ds.feature_names.index("city")]
    np.testing.assert_allclose(col[:3], [0.5, 0.5, 0.25])
    assert np.isnan(col[3])

    # Unseen category -> 0.0 (zero training frequency), missing stays NaN.
    df_new = df.copy()
    df_new.loc[0, "city"] = "kyoto"
    ds_new = RepLeafDataset(df_new, metadata=ds.metadata)
    col_new = ds_new.get_raw_features()[:, ds.feature_names.index("city")]
    assert col_new[0] == 0.0


def test_frequency_encoding_roundtrips_metadata():
    from repleafgbm.data.metadata import FeatureMetadata

    df = pd.DataFrame({"num1": [1.0, 2.0], "city": ["a", "b"]})
    ds = RepLeafDataset(df, frequency_encoded_features=["city"])
    restored = FeatureMetadata.from_dict(ds.metadata.to_dict())
    assert restored.frequency_maps == ds.metadata.frequency_maps


def test_frequency_and_categorical_overlap_rejected():
    df = pd.DataFrame({"city": ["a", "b"]})
    with pytest.raises(ValueError, match="both categorical"):
        RepLeafDataset(
            df, categorical_features=["city"], frequency_encoded_features=["city"]
        )


def test_uncastable_numerical_column_message():
    df = pd.DataFrame({"num1": [1.0, 2.0], "messy": ["1.5", "oops"]})
    with pytest.raises(ValueError, match="messy.*categorical_features"):
        RepLeafDataset(df, numerical_features=["num1", "messy"], categorical_features=[])


def test_high_cardinality_warning():
    df = pd.DataFrame({"id": [f"u{i}" for i in range(300)]})
    with pytest.warns(UserWarning, match="300 categories"):
        RepLeafDataset(df)


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
