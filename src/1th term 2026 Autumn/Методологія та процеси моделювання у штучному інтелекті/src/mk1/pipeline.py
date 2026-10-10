"""One end-to-end pipeline, configured by a Variant, for every method comparison.

The order of the steps is what keeps the evaluation honest: every step that
learns from data — group means, imputers, encoders, scalers, samplers — sits
inside the pipeline, so cross-validation refits it on each training fold and
the held-out fold is only ever transformed.
"""

from dataclasses import dataclass

import pandas as pd
from imblearn import FunctionSampler
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from mk1.augmentation import NOISE_SIGMA, drop_outliers, make_sampler, smotenc_columns
from mk1.config import SEED
from mk1.data import NUMERIC
from mk1.features import AGGREGATE_NUMERIC, DOMAIN_NUMERIC, FeatureEngineer
from mk1.preprocessing import CategoricalImputer, make_preprocessor

MODELS = ("logreg", "hgb")


@dataclass(frozen=True)
class Variant:
    """
    One combination of the compared preprocessing and augmentation choices.

    The defaults are the reference configuration: each experiment changes one
    field of it, so a difference in the metrics is attributable to that field.
    """

    impute: str = "missing"
    high_card: str = "target"
    scaler: str = "yeo-johnson"
    outliers: str = "keep"
    domain: bool = True
    aggregates: bool = True
    polynomial: bool = False
    sampler: str = "none"
    noise_sigma: float = NOISE_SIGMA


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Keeps the listed columns, so later steps never see the unused raw ones."""

    def __init__(self, columns: tuple[str, ...]) -> None:
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "ColumnSelector":
        """Nothing to learn."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return the selected columns."""
        return X[list(self.columns)]


def used_columns(variant: Variant) -> tuple[str, ...]:
    """Every column the variant's preprocessing reads, categorical ones first."""
    numeric = [*NUMERIC]
    if variant.domain:
        numeric += DOMAIN_NUMERIC
    if variant.aggregates:
        numeric += AGGREGATE_NUMERIC
    return (*smotenc_columns(variant.domain), *numeric)


def make_model(name: str, balanced: bool) -> ClassifierMixin:
    """
    Build one of the two reference classifiers.

    A linear model reacts to every encoding and scaling choice; a tree ensemble
    is invariant to monotone transformations and shows which choices matter
    regardless of them.

    Args:
        name: "logreg" or "hgb".
        balanced: Weight the classes inversely to their frequency.

    Returns:
        An unfitted classifier.

    Raises:
        ValueError: If the name is unknown.
    """
    weight = "balanced" if balanced else None
    match name:
        case "logreg":
            return LogisticRegression(max_iter=3000, class_weight=weight)
        case "hgb":
            return HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=250,
                max_leaf_nodes=31,
                l2_regularization=1.0,
                early_stopping=False,
                class_weight=weight,
                random_state=SEED,
            )
    raise ValueError(f"Unknown model: {name}")


def build_pipeline(variant: Variant, model: str) -> Pipeline:
    """
    Assemble the full pipeline from raw cleaned columns to a classifier.

    Args:
        variant: The preprocessing and augmentation choices.
        model: One of MODELS.

    Returns:
        An unfitted imbalanced-learn Pipeline; its samplers act during fit only.
    """
    steps = [
        (
            "features",
            FeatureEngineer(domain=variant.domain, aggregates=variant.aggregates),
        ),
        ("select", ColumnSelector(used_columns(variant))),
        ("impute", CategoricalImputer(strategy=variant.impute)),
    ]
    if variant.outliers != "keep":
        steps.append(
            (
                "outliers",
                FunctionSampler(
                    func=drop_outliers,
                    validate=False,
                    kw_args={"method": variant.outliers},
                ),
            )
        )
    # SMOTENC interpolates the raw categories, so it must run before encoding;
    # every other sampler works in the encoded numeric space after it.
    if variant.sampler == "smotenc":
        steps.append(("sampler", make_sampler("smotenc", domain=variant.domain)))
    steps.append(
        (
            "preprocess",
            make_preprocessor(
                scaler=variant.scaler,
                high_card=variant.high_card,
                domain=variant.domain,
                aggregates=variant.aggregates,
                polynomial=variant.polynomial,
            ),
        )
    )
    sampler = (
        make_sampler(variant.sampler, sigma=variant.noise_sigma)
        if variant.sampler != "smotenc"
        else None
    )
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(
        ("model", make_model(model, balanced=variant.sampler == "class-weight"))
    )
    return Pipeline(steps)
