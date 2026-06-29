"""Pluggable dataset-suite registry for the fair leaderboard.

The old ``openml_suite.py`` hardcoded its dataset list as a module-level tuple,
so swapping in a recognized suite (Grinsztajn 2022) or adding datasets meant
editing the runner. This registry makes a suite a named, declarative object:

* :class:`DatasetSpec` — one dataset (OpenML id or a builtin/synthetic source,
  task, optional target transform).
* :class:`SuiteSpec` — a named list of datasets with a ``--quick`` subset.
* :func:`load` — a generic loader (OpenML / sklearn builtin / offline synthetic)
  reusing ``openml_suite``'s fetch + ``clean_features`` so every model still sees
  the same ordinal-encoded matrix.

The ``grinsztajn_*`` suites are populated from the ``literature-scout`` note that
pins the exact OpenML benchmark-suite ids (see ``docs/research/``). Until then the
``legacy`` suite (the prior 9 datasets) and the offline ``synthetic`` suite keep
the pipeline runnable and CI-testable without network.

Lives under ``benchmarks/`` only; never imported by the library (``src/``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetSpec:
    """One benchmark dataset.

    ``openml_id`` is the OpenML ``data_id`` (the reproducibility version anchor);
    ``None`` means a non-OpenML source resolved by ``name`` — ``"california"``
    (sklearn builtin) or any ``"synthetic*"`` generator.
    """

    name: str
    openml_id: int | None
    task: str  # "regression" | "binary" | "multiclass"
    target_transform: str | None = None  # None | "log1p"


@dataclass(frozen=True)
class SuiteSpec:
    name: str
    datasets: tuple[DatasetSpec, ...]
    quick: tuple[str, ...] = ()
    description: str = ""

    def select(self, names: list[str] | None = None,
               quick: bool = False) -> tuple[DatasetSpec, ...]:
        ds = self.datasets
        if quick and self.quick:
            keep = set(self.quick)
            ds = tuple(d for d in ds if d.name in keep)
        if names:
            keep = set(names)
            ds = tuple(d for d in ds if d.name in keep)
        return ds


# --------------------------------------------------------------------------- #
# Suites
# --------------------------------------------------------------------------- #
# The prior openml_suite.py datasets, kept for continuity / A-B against history.
LEGACY = SuiteSpec(
    name="legacy",
    description="The original 9-dataset OpenML approximation (pre-overhaul).",
    datasets=(
        DatasetSpec("california", None, "regression"),
        DatasetSpec("house_sales", 42731, "regression", "log1p"),
        DatasetSpec("diamonds", 42225, "regression", "log1p"),
        DatasetSpec("wine_quality", 287, "regression"),
        DatasetSpec("credit_g", 31, "binary"),
        DatasetSpec("phoneme", 1489, "binary"),
        DatasetSpec("adult", 1590, "binary"),
        DatasetSpec("wine", 187, "multiclass"),
        DatasetSpec("vehicle", 54, "multiclass"),
    ),
    quick=("california", "diamonds", "credit_g", "phoneme", "wine"),
)

# Offline synthetic suite — no network, for CI smoke + the --quick e2e gate.
SYNTHETIC = SuiteSpec(
    name="synthetic",
    description="Offline generators (no OpenML download); for smoke tests.",
    datasets=(
        DatasetSpec("synthetic_reg", None, "regression"),
        DatasetSpec("synthetic_bin", None, "binary"),
        DatasetSpec("synthetic_multi", None, "multiclass"),
    ),
    quick=("synthetic_reg", "synthetic_bin"),
)

# Grinsztajn et al. 2022 "tabular benchmark" — the recognized suite cited by the
# paper. OpenML study ids 336/337/335/334, member data_ids verified by
# literature-scout (docs/research/2026-06-26-grinsztajn-suite-and-multioutput-datasets.md).
# Protocol: no target log-transform; the classification suites are binary.
def _specs(rows, task):
    return tuple(DatasetSpec(name, did, task) for did, name in rows)


GRINSZTAJN_NUM_REG = SuiteSpec(
    name="grinsztajn_num_reg",
    description="Grinsztajn 2022 numerical regression (OpenML study 336, 19 datasets).",
    datasets=_specs([
        (44132, "cpu_act"), (44133, "pol"), (44134, "elevators"),
        (44136, "wine_quality"), (44137, "Ailerons"), (44138, "houses"),
        (44139, "house_16H"), (44140, "diamonds"), (44141, "Brazilian_houses"),
        (44142, "Bike_Sharing_Demand"), (44143, "nyc-taxi-green-dec-2016"),
        (44144, "house_sales"), (44145, "sulfur"), (44146, "medical_charges"),
        (44147, "MiamiHousing2016"), (44148, "superconduct"), (45032, "yprop_4_1"),
        (45033, "abalone"), (45034, "delays_zurich_transport"),
    ], "regression"),
    quick=("abalone", "cpu_act"),
)

GRINSZTAJN_NUM_CLS = SuiteSpec(
    name="grinsztajn_num_cls",
    description="Grinsztajn 2022 numerical (binary) classification "
                "(OpenML study 337, 16 datasets).",
    datasets=_specs([
        (44089, "credit"), (44120, "electricity"), (44121, "covertype"),
        (44122, "pol"), (44123, "house_16H"), (44125, "MagicTelescope"),
        (44126, "bank-marketing"), (44128, "MiniBooNE"), (44129, "Higgs"),
        (44130, "eye_movements"), (45019, "Bioresponse"),
        (45020, "default-of-credit-card-clients"), (45021, "jannis"),
        (45022, "Diabetes130US"), (45026, "heloc"), (45028, "california"),
    ], "binary"),
    quick=("credit", "california"),
)

GRINSZTAJN_CAT_REG = SuiteSpec(
    name="grinsztajn_cat_reg",
    description="Grinsztajn 2022 categorical regression (OpenML study 335, 17 datasets).",
    datasets=_specs([
        (44055, "analcatdata_supreme"), (44056, "visualizing_soil"),
        (44059, "diamonds"), (44061, "Mercedes_Benz_Greener_Manufacturing"),
        (44062, "Brazilian_houses"), (44063, "Bike_Sharing_Demand"),
        (44065, "nyc-taxi-green-dec-2016"), (44066, "house_sales"),
        (44068, "particulate-matter-ukair-2017"),
        (44069, "SGEMM_GPU_kernel_performance"), (45041, "topo_2_1"),
        (45042, "abalone"), (45043, "seattlecrime6"),
        (45045, "delays_zurich_transport"), (45046, "Allstate_Claims_Severity"),
        (45047, "Airlines_DepDelay_1M"), (45048, "medical_charges"),
    ], "regression"),
    quick=("analcatdata_supreme", "visualizing_soil"),
)

GRINSZTAJN_CAT_CLS = SuiteSpec(
    name="grinsztajn_cat_cls",
    description="Grinsztajn 2022 categorical (binary) classification "
                "(OpenML study 334, 7 datasets).",
    datasets=_specs([
        (44156, "electricity"), (44157, "eye_movements"), (44159, "covertype"),
        (45035, "albert"), (45036, "default-of-credit-card-clients"),
        (45038, "road-safety"), (45039, "compas-two-years"),
    ], "binary"),
    quick=("compas-two-years", "eye_movements"),
)

#: Registry. The Grinsztajn suites are the recognized leaderboard anchor; legacy
#: and synthetic stay for continuity / offline CI. Order matters only for
#: ``find()`` (first match wins), so the simple regression datasets — used by the
#: router-extraction study via bare names like "california" — come first.
SUITES: dict[str, SuiteSpec] = {s.name: s for s in (
    LEGACY, SYNTHETIC,
    GRINSZTAJN_NUM_REG, GRINSZTAJN_NUM_CLS, GRINSZTAJN_CAT_REG, GRINSZTAJN_CAT_CLS,
)}


def register_suite(spec: SuiteSpec) -> None:
    """Add or replace a suite in the registry (used to wire in Grinsztajn ids)."""
    SUITES[spec.name] = spec


def get_suite(name: str) -> SuiteSpec:
    if name not in SUITES:
        raise KeyError(f"unknown suite {name!r}; have {sorted(SUITES)}")
    return SUITES[name]


def find(name: str) -> DatasetSpec:
    """Return the first :class:`DatasetSpec` named ``name`` across all suites."""
    for suite in SUITES.values():
        for ds in suite.datasets:
            if ds.name == name:
                return ds
    raise KeyError(f"unknown dataset {name!r}")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _synthetic(name: str, n_rows: int, seed: int) -> tuple[pd.DataFrame, np.ndarray, str]:
    """Offline dataset from benchmarks.common generators. No network."""
    from benchmarks.common import synthetic_tabular

    rng = np.random.default_rng(seed)
    n_features = 10
    X, signal = synthetic_tabular(n_rows, n_features, rng)
    Xdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    if name == "synthetic_reg":
        y = signal + 0.1 * rng.normal(size=n_rows)
        return Xdf, y.astype(np.float64), "regression"
    if name == "synthetic_bin":
        y = (signal > np.median(signal)).astype(int)
        return Xdf, y, "binary"
    if name == "synthetic_multi":
        edges = np.quantile(signal, [1 / 3, 2 / 3])
        y = np.digitize(signal, edges)
        return Xdf, y, "multiclass"
    raise ValueError(f"unknown synthetic dataset {name!r}")


def load(spec: DatasetSpec, n_rows: int = 4000, seed: int = 0):
    """Return ``(X DataFrame, y ndarray, categorical_columns)`` for a dataset.

    Reuses ``openml_suite``'s certifi-aware fetch and ``clean_features`` so the
    leaderboard feeds every model the same ordinal-encoded matrix. ``n_rows`` /
    ``seed`` only affect the offline synthetic datasets.
    """
    from benchmarks.openml_suite import _fetch, clean_features
    from sklearn.preprocessing import LabelEncoder

    if spec.openml_id is None and spec.name.startswith("synthetic"):
        X, y, _ = _synthetic(spec.name, n_rows, seed)
        Xc, cats = clean_features(X)
        return Xc, y, cats

    if spec.openml_id is None:  # sklearn builtin (california)
        from sklearn.datasets import fetch_california_housing

        d = fetch_california_housing(as_frame=True)
        X, y = d.data, d.target.to_numpy(np.float64)
    else:
        d = _fetch(spec.openml_id)
        X = d.data
        if spec.task == "regression":
            y = d.target.to_numpy(np.float64)
        else:
            labels = d.target.astype(str).str.strip(" '\"")
            y = LabelEncoder().fit_transform(labels)

    if spec.task == "regression" and spec.target_transform == "log1p":
        y = np.log1p(y)
    Xc, cats = clean_features(X)
    return Xc, y, cats
