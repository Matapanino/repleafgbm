"""Feature metadata: names, types, and categorical encodings.

FeatureMetadata is the single source of truth about how raw input columns are
interpreted. It is created when a :class:`~repleafgbm.data.RepLeafDataset` is
built from training data, stored inside fitted models, and re-applied to new
data at prediction time so that train/predict preprocessing always matches.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FeatureMetadata:
    """Describes the columns of the raw feature matrix.

    Attributes:
        feature_names: All feature names in column order.
        numerical_features: Names of numerical columns.
        categorical_features: Names of categorical columns.
        category_maps: For each categorical feature, the ordered list of known
            category values (stored as strings). A category's ordinal code is
            its index in this list; unseen categories and missing values map
            to NaN in the raw feature matrix. For ``pandas.Categorical``
            columns the declared category order is preserved (including
            categories not observed in the training sample).
        frequency_maps: For each frequency-encoded feature, the mapping from
            category value (as string) to its training-set frequency
            (proportion of rows). Frequency-encoded features are *numerical*
            downstream: they get threshold splits and are visible to the
            encoder. Unseen categories encode to 0.0 (zero training
            frequency); missing values stay NaN.
    """

    feature_names: list[str]
    numerical_features: list[str]
    categorical_features: list[str]
    category_maps: dict[str, list[str]] = field(default_factory=dict)
    frequency_maps: dict[str, dict[str, float]] = field(default_factory=dict)
    leaf_only_features: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        known = set(self.numerical_features) | set(self.categorical_features)
        unknown = [f for f in self.feature_names if f not in known]
        if unknown:
            raise ValueError(
                f"Features {unknown} are neither numerical nor categorical. "
                "Pass them via numerical_features or categorical_features."
            )
        bad_freq = [
            f
            for f in self.frequency_maps
            if f not in set(self.numerical_features) or f in set(self.categorical_features)
        ]
        if bad_freq:
            raise ValueError(
                f"Frequency-encoded features {bad_freq} must be listed as "
                "numerical (they are routed with threshold splits), not "
                "categorical."
            )
        missing_leaf_only = [f for f in self.leaf_only_features if f not in self.feature_names]
        if missing_leaf_only:
            raise ValueError(f"leaf-only features {missing_leaf_only} are not input columns")
        duplicate_leaf_only = len(set(self.leaf_only_features)) != len(
            self.leaf_only_features
        )
        if duplicate_leaf_only:
            raise ValueError("leaf-only features must not contain duplicates")
        categorical_leaf_only = [
            f for f in self.leaf_only_features if f in set(self.categorical_features)
        ]
        if categorical_leaf_only:
            raise ValueError(
                f"leaf-only features {categorical_leaf_only} are categorical; "
                "leaf-only columns must be finite numerical columns"
            )

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @property
    def numerical_indices(self) -> list[int]:
        """Ordered encoder-input indices: regular numerical, then leaf-only."""
        position = {f: i for i, f in enumerate(self.feature_names)}
        return [position[f] for f in self.ordered_numerical_features]

    @property
    def ordered_numerical_features(self) -> list[str]:
        """Encoder block names, with leaf-only columns appended as declared."""
        leaf_only = set(self.leaf_only_features)
        regular = [
            f
            for f in self.feature_names
            if f in set(self.numerical_features) and f not in leaf_only
        ]
        return regular + list(self.leaf_only_features)

    @property
    def routing_feature_names(self) -> list[str]:
        """Input columns available to tree split search, in raw order."""
        leaf_only = set(self.leaf_only_features)
        return [f for f in self.feature_names if f not in leaf_only]

    @property
    def routing_indices(self) -> list[int]:
        """Column indices of features available to tree split search."""
        leaf_only = set(self.leaf_only_features)
        return [i for i, f in enumerate(self.feature_names) if f not in leaf_only]

    @property
    def routing_categorical_indices(self) -> list[int]:
        """Categorical indices relative to the routing feature matrix."""
        categorical = set(self.categorical_features)
        return [
            i for i, f in enumerate(self.routing_feature_names) if f in categorical
        ]

    @property
    def categorical_indices(self) -> list[int]:
        """Column indices of categorical features in the raw feature matrix."""
        return [i for i, f in enumerate(self.feature_names) if f in set(self.categorical_features)]

    def to_dict(self) -> dict:
        d = {
            "feature_names": list(self.feature_names),
            "numerical_features": list(self.numerical_features),
            "categorical_features": list(self.categorical_features),
            "category_maps": {k: list(v) for k, v in self.category_maps.items()},
        }
        # Only written when used, so ordinal-only models keep the exact
        # format that older builds (format_version 3) can read.
        if self.frequency_maps:
            d["frequency_maps"] = {k: dict(v) for k, v in self.frequency_maps.items()}
        if self.leaf_only_features:
            d["leaf_only_features"] = list(self.leaf_only_features)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> FeatureMetadata:
        return cls(
            feature_names=list(d["feature_names"]),
            numerical_features=list(d["numerical_features"]),
            categorical_features=list(d["categorical_features"]),
            category_maps={k: list(v) for k, v in d.get("category_maps", {}).items()},
            frequency_maps={k: dict(v) for k, v in d.get("frequency_maps", {}).items()},
            leaf_only_features=list(d.get("leaf_only_features", [])),
        )
