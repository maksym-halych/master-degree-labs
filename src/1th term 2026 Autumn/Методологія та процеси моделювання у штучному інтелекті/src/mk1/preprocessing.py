"""Imputation, encoding and scaling building blocks (task 3).

Each step is a scikit-learn transformer, so a cross-validation fold or the
final fit learns its statistics from training rows only.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
    TargetEncoder,
)

from mk1.config import SEED
from mk1.data import AGE_ORDER, CATEGORICAL, HIGH_CARD, LOW_CARD, NUMERIC
from mk1.features import AGGREGATE_NUMERIC, DIAG_GROUPS, DOMAIN_NUMERIC, MISSING

# Categories rarer than this are pooled into one "infrequent" column by the
# One-Hot variant for high-cardinality columns; without it the diagnosis codes
# alone would add some 2000 mostly empty columns.
ONEHOT_MIN_FREQUENCY = 50


class CategoricalImputer(BaseEstimator, TransformerMixin):
    """
    Fills missing categorical values.

    Args:
        strategy: "missing" adds an explicit category, "mode" takes the most
            frequent training value, "knn" predicts the value from the numeric
            features of the row's nearest training neighbours.
        columns: Categorical columns to fill.
        context: Numeric columns the "knn" strategy measures distance on.
        n_neighbors: Neighbours consulted by the "knn" strategy.
    """

    def __init__(
        self,
        strategy: str = "missing",
        columns: tuple[str, ...] = tuple(CATEGORICAL),
        context: tuple[str, ...] = tuple(NUMERIC),
        n_neighbors: int = 15,
    ) -> None:
        self.strategy = strategy
        self.columns = columns
        self.context = context
        self.n_neighbors = n_neighbors

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "CategoricalImputer":
        """Learn the fill value, or the neighbour model, of each column."""
        self.modes_ = {c: X[c].mode().iloc[0] for c in self.columns}
        self.models_ = {}
        if self.strategy == "knn":
            self.scaler_ = StandardScaler().fit(X[list(self.context)])
            Z = self.scaler_.transform(X[list(self.context)])
            for c in self.columns:
                observed = X[c].notna().to_numpy()
                if observed.all():
                    continue
                model = KNeighborsClassifier(n_neighbors=self.n_neighbors)
                self.models_[c] = model.fit(Z[observed], X[c].to_numpy()[observed])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of X with no missing value left in the columns."""
        out = X.copy()
        if self.strategy == "knn":
            Z = self.scaler_.transform(X[list(self.context)])
        for c in self.columns:
            missing = out[c].isna().to_numpy()
            if not missing.any():
                continue
            if self.strategy == "missing":
                out[c] = out[c].fillna(MISSING)
            elif self.strategy == "mode" or c not in self.models_:
                out[c] = out[c].fillna(self.modes_[c])
            else:
                values = out[c].to_numpy(dtype=object, copy=True)
                values[missing] = self.models_[c].predict(Z[missing])
                out[c] = values
        return out


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Replaces each category with its share of the training rows."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FrequencyEncoder":
        """Count category shares per column."""
        X = pd.DataFrame(X)
        self.frequencies_ = [X[c].value_counts(normalize=True) for c in X.columns]
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Map categories to shares; a category unseen in training gets 0."""
        X = pd.DataFrame(X)
        columns = [
            X.iloc[:, i].map(freq).astype(float).fillna(0.0).to_numpy()
            for i, freq in enumerate(self.frequencies_)
        ]
        return np.column_stack(columns)

    def get_feature_names_out(
        self, input_features: list[str] | None = None
    ) -> np.ndarray:
        """Keep the input column names, as the encoding is one-to-one."""
        return np.asarray(input_features, dtype=object)


class ShiftedBoxCox(BaseEstimator, TransformerMixin):
    """
    Box-Cox for columns that hold zeros or negative values.

    Box-Cox is defined for strictly positive values only, while every count
    here starts at zero and the aggregate features are centred around it. Each
    column is shifted so its training minimum becomes 1; a test value below
    that minimum is clipped to stay inside the domain.
    """

    def fit(self, X: np.ndarray, y: pd.Series | None = None) -> "ShiftedBoxCox":
        """Learn the per-column shift and the Box-Cox lambdas."""
        X = np.asarray(X, dtype=float)
        self.shift_ = 1.0 - X.min(axis=0)
        self.power_ = PowerTransformer(method="box-cox").fit(X + self.shift_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Shift, clip at the training minimum, then apply Box-Cox."""
        shifted = np.maximum(np.asarray(X, dtype=float) + self.shift_, 1.0)
        return self.power_.transform(shifted)


def make_scaler(name: str) -> TransformerMixin:
    """
    Build one of the compared numeric transformations.

    Args:
        name: "standard", "robust", "yeo-johnson" or "box-cox".

    Returns:
        An unfitted transformer.

    Raises:
        ValueError: If the name is unknown.
    """
    match name:
        case "standard":
            return StandardScaler()
        case "robust":
            return RobustScaler()
        case "yeo-johnson":
            return PowerTransformer(method="yeo-johnson")
        case "box-cox":
            return ShiftedBoxCox()
    raise ValueError(f"Unknown scaler: {name}")


def make_high_card_encoder(name: str) -> TransformerMixin:
    """
    Build the encoder for the high-cardinality columns.

    Args:
        name: "target", "frequency" or "onehot".

    Returns:
        An unfitted encoder.

    Raises:
        ValueError: If the name is unknown.
    """
    match name:
        case "target":
            # Cross-fitting is what keeps target encoding honest: each training
            # row is encoded with means learned on the other folds, never with
            # its own label.
            folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
            return TargetEncoder(target_type="binary", cv=folds)
        case "frequency":
            return FrequencyEncoder()
        case "onehot":
            return OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                min_frequency=ONEHOT_MIN_FREQUENCY,
                sparse_output=False,
                dtype=np.float32,
            )
    raise ValueError(f"Unknown high-cardinality encoder: {name}")


def make_preprocessor(
    scaler: str = "yeo-johnson",
    high_card: str = "target",
    domain: bool = True,
    aggregates: bool = True,
    polynomial: bool = False,
) -> ColumnTransformer:
    """
    Assemble the column-wise preprocessing of a cleaned, imputed frame.

    Args:
        scaler: Numeric transformation, see make_scaler().
        high_card: Encoder of the HIGH_CARD columns, see make_high_card_encoder().
        domain: Whether the domain features of FeatureEngineer are present.
        aggregates: Whether its aggregate features are present.
        polynomial: Add degree-2 terms of the base numeric features.

    Returns:
        An unfitted ColumnTransformer producing a dense float matrix.
    """
    numeric = [*NUMERIC]
    low_card = [*LOW_CARD]
    if domain:
        numeric += DOMAIN_NUMERIC
        low_card += DIAG_GROUPS
    if aggregates:
        numeric += AGGREGATE_NUMERIC

    blocks = []
    if polynomial:
        # Squares and pairwise products of the base counts, built on the
        # already transformed values so that one outlier cannot dominate them.
        poly = Pipeline(
            [
                ("scale", make_scaler(scaler)),
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ]
        )
        blocks.append(("poly", poly, NUMERIC))
        numeric = [c for c in numeric if c not in NUMERIC]
    # Ordinal codes and encoded shares or means come out on scales of their own;
    # standardising them keeps a linear model's penalty equal across blocks.
    age = Pipeline(
        [
            ("ordinal", OrdinalEncoder(categories=[AGE_ORDER])),
            ("scale", StandardScaler()),
        ]
    )
    high = make_high_card_encoder(high_card)
    if high_card != "onehot":
        high = Pipeline([("encode", high), ("scale", StandardScaler())])
    blocks += [
        ("num", make_scaler(scaler), numeric),
        ("age", age, ["age"]),
        (
            "low",
            OneHotEncoder(
                handle_unknown="ignore", sparse_output=False, dtype=np.float32
            ),
            low_card,
        ),
        ("high", high, HIGH_CARD),
    ]
    return ColumnTransformer(blocks, remainder="drop", sparse_threshold=0.0)
