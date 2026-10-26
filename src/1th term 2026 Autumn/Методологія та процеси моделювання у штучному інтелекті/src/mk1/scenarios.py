"""The data scenarios and classifiers that lab 2 compares.

Every scenario starts from the same cleaned rows and the same train/test split
as lab 1, so the scenarios differ only in what is done to the features. The
row-level cleaning itself (one encounter per patient, no deaths or hospice
discharges) is not a scenario choice: without it a patient could sit on both
sides of the split, and every score would be inflated by leakage.

Scenario labels stay in Ukrainian: they are the tick labels and legends of
report figures.
"""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from mk1.config import SEED
from mk1.data import AGE_ORDER, CATEGORICAL, NUMERIC
from mk1.evaluation import FINALS
from mk1.pipeline import ColumnSelector, Variant, build_pipeline, make_model
from mk1.preprocessing import ONEHOT_MIN_FREQUENCY, CategoricalImputer

MODELS = ("logreg", "rf", "hgb")

# Minimum number of training rows in a leaf of the random forest. Fully grown
# trees (a leaf of one row) memorise the training set; the overfitting stage
# sweeps this value to show it.
RF_MIN_LEAF = 10

# Lab 1's final configuration for the linear model: "Missing" category for
# absent values, target encoding of the high-cardinality columns, RobustScaler,
# domain and aggregate features. The tree ensembles were indifferent to the
# encoding and scaling choices within the fold-to-fold spread, so one
# configuration serves all three models and the scenario stays a property of
# the data alone.
CLEANED: Variant = FINALS["logreg"]


@dataclass(frozen=True)
class Scenario:
    """
    One way of preparing the data for the classifiers.

    Args:
        key: Short identifier used in the metric files.
        label: Name shown in the report figures.
        short: Abbreviated name for the tick labels of crowded figures.
        variant: Lab 1 pipeline configuration, or None for the minimal
            preprocessing of minimal_steps().
    """

    key: str
    label: str
    short: str
    variant: Variant | None


SCENARIOS = (
    Scenario("A", "А: мінімальне очищення", "А", None),
    Scenario("B", "Б: повне очищення", "Б", CLEANED),
    Scenario(
        "C-smote", "В: очищення + SMOTE", "В (SMOTE)", replace(CLEANED, sampler="smote")
    ),
    Scenario(
        "C-adasyn",
        "В: очищення + ADASYN",
        "В (ADASYN)",
        replace(CLEANED, sampler="adasyn"),
    ),
)
SCENARIO_BY_KEY = {s.key: s for s in SCENARIOS}


def minimal_steps() -> list[tuple[str, object]]:
    """
    The least preprocessing that lets scikit-learn accept the data.

    The most frequent value fills every gap, every categorical column is
    one-hot encoded and the counts go in unscaled. No engineered features.
    Rare categories are still pooled: without that, the diagnosis codes alone
    would add some 2000 columns, most of them empty.

    Returns:
        Pipeline steps from the cleaned frame to a dense numeric matrix.
    """
    onehot = OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        min_frequency=ONEHOT_MIN_FREQUENCY,
        sparse_output=False,
        dtype=np.float32,
    )
    encode = ColumnTransformer(
        [
            ("num", "passthrough", NUMERIC),
            ("age", OrdinalEncoder(categories=[AGE_ORDER]), ["age"]),
            ("cat", onehot, CATEGORICAL),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    return [
        ("select", ColumnSelector((*CATEGORICAL, "age", *NUMERIC))),
        ("impute", CategoricalImputer(strategy="mode")),
        ("encode", encode),
    ]


def make_classifier(name: str) -> ClassifierMixin:
    """
    Build one of the compared classifiers, without class weights.

    Args:
        name: "logreg", "rf" or "hgb". The first and the last are lab 1's
            reference models with the same hyperparameters.

    Returns:
        An unfitted classifier.

    Raises:
        ValueError: If the name is unknown.
    """
    match name:
        case "logreg" | "hgb":
            return make_model(name, balanced=False)
        case "rf":
            return RandomForestClassifier(
                n_estimators=300,
                max_features="sqrt",
                min_samples_leaf=RF_MIN_LEAF,
                n_jobs=-1,
                random_state=SEED,
            )
    raise ValueError(f"Unknown model: {name}")


def build_scenario(scenario: Scenario, model: str) -> Pipeline:
    """
    Assemble the pipeline of one scenario ending in one classifier.

    Args:
        scenario: The data preparation to apply.
        model: One of MODELS.

    Returns:
        An unfitted imbalanced-learn Pipeline; its sampler, if any, acts
        during fit only, so validation and test rows are never resampled.
    """
    if scenario.variant is None:
        steps = minimal_steps()
    else:
        # Lab 1's preprocessing and sampling steps, unchanged. build_pipeline
        # needs a model name to be complete; its classifier is dropped here.
        steps = build_pipeline(scenario.variant, "logreg").steps[:-1]
    return Pipeline([*steps, ("model", make_classifier(model))])


def fit_on_matrix(
    pipe: Pipeline, X: pd.DataFrame, y: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit a pipeline and return the matrix its classifier was trained on.

    Equivalent to pipe.fit(X, y), but the preprocessed and resampled training
    matrix stays available. An over-sampler appends its synthetic rows after
    the real ones, so the first len(X) rows of the matrix are the real
    patients and the rest, if any, are synthetic.

    Args:
        pipe: An unfitted pipeline from build_scenario().
        X: Training features.
        y: Training labels.

    Returns:
        The training matrix and its labels, real rows first.
    """
    head = pipe[:-1]
    if hasattr(head.steps[-1][1], "fit_resample"):
        Xt, yt = head.fit_resample(X, y)
    else:
        Xt, yt = head.fit_transform(X, y), y
    Xt, yt = np.asarray(Xt), np.asarray(yt)
    pipe.named_steps["model"].fit(Xt, yt)
    return Xt, yt
