"""Shared test fixtures: small, deterministic synthetic datasets."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def regression_data():
    """Piecewise regime on x0 with strong linear structure inside regimes.

    Designed so that representation-conditioned leaves have an edge over
    constant leaves: a few raw splits isolate the regimes, and within each
    regime the target is linear in x1/x2.
    """
    rng = np.random.default_rng(7)
    n = 500
    X = rng.normal(size=(n, 4))
    y = (
        np.where(X[:, 0] > 0.0, 3.0, -2.0)
        + 2.0 * X[:, 1]
        - 1.0 * X[:, 2]
        + rng.normal(0.0, 0.05, n)
    )
    return X[:350], y[:350], X[350:], y[350:]


@pytest.fixture
def classification_data():
    rng = np.random.default_rng(11)
    n = 500
    X = rng.normal(size=(n, 4))
    logit = 2.0 * X[:, 0] + 1.5 * X[:, 1] - X[:, 2]
    y = (logit + rng.normal(0.0, 0.5, n) > 0).astype(int)
    return X[:350], y[:350], X[350:], y[350:]
