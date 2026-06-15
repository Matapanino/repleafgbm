"""RepLeafGBM: Representation-enhanced Leaf Gradient Boosting Machine.

RepLeafGBM is not a neural network inside a tree, nor a tree over embeddings.
It is a boosted ensemble of raw-feature routers with representation-conditioned
local predictors.
"""

from repleafgbm.classifier import RepLeafClassifier
from repleafgbm.core.metrics import make_metric
from repleafgbm.core.objectives import Huber, PoissonRegression, Quantile
from repleafgbm.data import RepLeafDataset
from repleafgbm.regressor import RepLeafRegressor

__version__ = "1.0.1"

__all__ = [
    "RepLeafRegressor",
    "RepLeafClassifier",
    "RepLeafDataset",
    "make_metric",
    "Huber",
    "Quantile",
    "PoissonRegression",
    "__version__",
]
