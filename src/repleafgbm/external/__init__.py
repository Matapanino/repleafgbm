"""External GBM integrations: the *external_model* mode of
docs/backend_strategy.md.

External libraries are trained as independent base models whose outputs
(predictions, leaf indices) become features for RepLeafGBM — model
diversity, not a wrapper. Guardrails:

* Nothing in the native path imports this package.
* Importing ``repleafgbm.external`` itself is safe without any GBM installed;
  each dependency is checked at call time with a clear message. The
  ``[external]`` extra installs **LightGBM** (the base for ``router_extraction``);
  XGBoost and CatBoost ship via the ``[bench]`` extra (or ``pip install
  xgboost`` / ``pip install catboost`` individually).
"""

from repleafgbm.external.catboost_model import CatBoostExternalModel
from repleafgbm.external.features import augment_features, external_feature_frame
from repleafgbm.external.lightgbm_model import LightGBMExternalModel
from repleafgbm.external.oof import oof_predictions
from repleafgbm.external.router_extraction import (
    RouterExtractionClassifier,
    RouterExtractionRegressor,
    extract_routes,
)
from repleafgbm.external.xgboost_model import XGBoostExternalModel

__all__ = [
    "LightGBMExternalModel",
    "XGBoostExternalModel",
    "CatBoostExternalModel",
    "oof_predictions",
    "external_feature_frame",
    "augment_features",
    "extract_routes",
    "RouterExtractionRegressor",
    "RouterExtractionClassifier",
]
