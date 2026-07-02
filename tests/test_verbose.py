"""Tests for per-round eval logging (``verbose``)."""

import re

import numpy as np
import pytest

from repleafgbm import RepLeafClassifier, RepLeafDataset, RepLeafRegressor

ROUND_LINE = re.compile(r"^\[(\d+)\]\t(valid_\d+'s [\w().]+: \d+\.\d{6}(\t)?)+$")


@pytest.fixture
def reg_split():
    rng = np.random.default_rng(0)
    n = 400
    X = rng.normal(size=(n, 5))
    y = X[:, 0] * 2.0 + np.sin(X[:, 1]) + rng.normal(0.0, 0.1, n)
    return X[:300], y[:300], X[300:], y[300:]


def _fit_regressor(reg_split, capsys, **kwargs):
    Xtr, ytr, Xva, yva = reg_split
    params = dict(
        n_estimators=10,
        num_leaves=8,
        min_samples_leaf=5,
        leaf_model="constant",
        random_state=42,
    )
    params.update(kwargs)
    model = RepLeafRegressor(**params)
    train = RepLeafDataset(Xtr, ytr)
    model.fit(train, eval_set=[RepLeafDataset(Xva, yva, metadata=train.metadata)])
    return model, capsys.readouterr().out


def test_default_is_silent(reg_split, capsys):
    _, out = _fit_regressor(reg_split, capsys)
    assert out == ""


def test_verbose_zero_is_silent(reg_split, capsys):
    _, out = _fit_regressor(reg_split, capsys, verbose=0)
    assert out == ""


def test_verbose_one_logs_every_round(reg_split, capsys):
    _, out = _fit_regressor(reg_split, capsys, verbose=1)
    lines = out.strip().splitlines()
    assert len(lines) == 10
    for i, line in enumerate(lines, start=1):
        assert ROUND_LINE.match(line), line
        assert line.startswith(f"[{i}]\t")


def test_verbose_period_is_respected(reg_split, capsys):
    _, out = _fit_regressor(reg_split, capsys, verbose=3)
    lines = out.strip().splitlines()
    assert [int(ln.split("]")[0][1:]) for ln in lines] == [3, 6, 9]


def test_logged_scores_match_evals_result(reg_split, capsys):
    model, out = _fit_regressor(reg_split, capsys, verbose=1)
    history = model.evals_result_["valid_0"]["rmse"]
    logged = [float(ln.rsplit(": ", 1)[1]) for ln in out.strip().splitlines()]
    assert logged == pytest.approx(history, abs=5e-7)


def test_verbose_without_eval_set_is_silent(reg_split, capsys):
    Xtr, ytr, *_ = reg_split
    model = RepLeafRegressor(
        n_estimators=5, num_leaves=8, min_samples_leaf=5, verbose=1, random_state=42
    )
    model.fit(Xtr, ytr)
    assert capsys.readouterr().out == ""


def test_two_eval_sets_share_one_line(reg_split, capsys):
    Xtr, ytr, Xva, yva = reg_split
    model = RepLeafRegressor(
        n_estimators=4,
        num_leaves=8,
        min_samples_leaf=5,
        leaf_model="constant",
        verbose=1,
        random_state=42,
    )
    train = RepLeafDataset(Xtr, ytr)
    model.fit(
        train,
        eval_set=[
            RepLeafDataset(Xva[:50], yva[:50], metadata=train.metadata),
            RepLeafDataset(Xva[50:], yva[50:], metadata=train.metadata),
        ],
    )
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 4
    assert "valid_0's rmse" in lines[0] and "valid_1's rmse" in lines[0]


def test_early_stopping_prints_best_iteration(capsys):
    rng = np.random.default_rng(3)
    n = 600
    X = rng.normal(size=(n, 5))
    y = X[:, 0] + 0.5 * X[:, 1] + rng.normal(0.0, 2.0, n)
    model = RepLeafRegressor(
        n_estimators=300,
        learning_rate=0.3,
        num_leaves=16,
        min_samples_leaf=5,
        leaf_model="constant",
        early_stopping_rounds=10,
        verbose=1,
        random_state=42,
    )
    train = RepLeafDataset(X[:400], y[:400])
    model.fit(
        train, eval_set=[RepLeafDataset(X[400:], y[400:], metadata=train.metadata)]
    )
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[-2] == "Early stopping, best iteration is:"
    assert lines[-1].startswith(f"[{model.best_iteration_}]\t")
    best = model.evals_result_["valid_0"]["rmse"][model.best_iteration_ - 1]
    assert float(lines[-1].rsplit(": ", 1)[1]) == pytest.approx(best, abs=5e-7)


def test_multiclass_early_stop_prints_best_iteration(capsys):
    rng = np.random.default_rng(5)
    n = 600
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] + rng.normal(0.0, 2.0, n) > 0).astype(int) + (
        X[:, 1] + rng.normal(0.0, 2.0, n) > 0.5
    ).astype(int)
    model = RepLeafClassifier(
        n_estimators=200,
        learning_rate=0.3,
        num_leaves=8,
        min_samples_leaf=5,
        leaf_model="constant",
        early_stopping_rounds=5,
        verbose=1,
        random_state=42,
    )
    train = RepLeafDataset(X[:400], y[:400])
    model.fit(
        train, eval_set=[RepLeafDataset(X[400:], y[400:], metadata=train.metadata)]
    )
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[-2] == "Early stopping, best iteration is:"
    assert lines[-1].startswith(f"[{model.best_iteration_}]\t")


def test_multioutput_early_stop_prints_best_iteration(capsys):
    rng = np.random.default_rng(6)
    n = 600
    X = rng.normal(size=(n, 4))
    Y = np.column_stack([X[:, 0], X[:, 1]]) + rng.normal(0.0, 2.0, (n, 2))
    model = RepLeafRegressor(
        n_estimators=200,
        learning_rate=0.3,
        num_leaves=8,
        min_samples_leaf=5,
        leaf_model="constant",
        early_stopping_rounds=5,
        verbose=1,
        random_state=42,
    )
    train = RepLeafDataset(X[:400], Y[:400])
    model.fit(
        train, eval_set=[RepLeafDataset(X[400:], Y[400:], metadata=train.metadata)]
    )
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[-2] == "Early stopping, best iteration is:"
    assert lines[-1].startswith(f"[{model.best_iteration_}]\t")


def test_multiclass_verbose(capsys):
    rng = np.random.default_rng(1)
    n = 300
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int) + (X[:, 2] > 0.5).astype(int)
    model = RepLeafClassifier(
        n_estimators=6,
        num_leaves=8,
        min_samples_leaf=5,
        leaf_model="constant",
        verbose=2,
        random_state=42,
    )
    train = RepLeafDataset(X[:200], y[:200])
    model.fit(
        train, eval_set=[RepLeafDataset(X[200:], y[200:], metadata=train.metadata)]
    )
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 3
    assert all("valid_0's multi_logloss" in ln for ln in lines)
    assert [int(ln.split("]")[0][1:]) for ln in lines] == [2, 4, 6]


def test_multioutput_verbose(capsys):
    rng = np.random.default_rng(2)
    n = 300
    X = rng.normal(size=(n, 4))
    Y = np.column_stack([X[:, 0] + X[:, 1], X[:, 2] - X[:, 3]])
    Y += rng.normal(0.0, 0.1, Y.shape)
    model = RepLeafRegressor(
        n_estimators=5,
        num_leaves=8,
        min_samples_leaf=5,
        leaf_model="constant",
        verbose=1,
        random_state=42,
    )
    train = RepLeafDataset(X[:200], Y[:200])
    model.fit(
        train, eval_set=[RepLeafDataset(X[200:], Y[200:], metadata=train.metadata)]
    )
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 5
    assert all(ROUND_LINE.match(ln) for ln in lines)
