"""Imputation compared on values whose truth is known (task 3).

The numeric features of this dataset are complete, so their imputers are
compared by hiding observed values on purpose — under MCAR (missing completely
at random) and under MAR (missingness driven by another feature) — and
measuring how close each method gets to what was hidden. The categorical
features have real gaps; for them the same masking measures how often the
hidden category is recovered.

Figure titles and axis labels stay in Ukrainian: they are report content, read
off the rendered figure, not code the reader of this module has to follow.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.experimental import (
    enable_iterative_imputer,  # noqa: F401 — registers IterativeImputer
)
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split

from mk1.artifacts import save_figure, save_json
from mk1.config import IMPUTATION_SAMPLE, LAB1, SEED
from mk1.data import AGE_ORDER, NUMERIC
from mk1.preprocessing import CategoricalImputer

log = logging.getLogger(__name__)

MASK_RATE = 0.20

# The MAR mechanism hides values more often the longer the stay, so this
# column itself always stays observed.
ANCHOR = "time_in_hospital"
MASKED = [c for c in NUMERIC if c != ANCHOR]

METHODS = ("median", "knn", "mice")
LABELS = {
    "median": "медіана",
    "knn": "KNNImputer (k = 5)",
    "mice": "IterativeImputer (MICE)",
}
COLORS = {"median": "grey", "knn": "steelblue", "mice": "darkorange"}

CATEGORICAL_TARGETS = [
    "race",
    "payer_code",
    "medical_specialty",
    "admission_type_id",
    "admission_source_id",
    "discharge_disposition_id",
]


def sample(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Stratified subsample: KNNImputer is quadratic in the number of rows."""
    part, _ = train_test_split(
        X, train_size=IMPUTATION_SAMPLE, stratify=y, random_state=SEED
    )
    return part.reset_index(drop=True)


def make_mask(
    X: pd.DataFrame, mechanism: str, rng: np.random.Generator
) -> pd.DataFrame:
    """
    Choose which cells of the MASKED columns to hide.

    Args:
        X: Complete numeric frame.
        mechanism: "mcar" hides every cell with the same probability; "mar"
            makes the probability grow linearly with the rank of ANCHOR, from
            0 for the shortest stays to twice MASK_RATE for the longest.
        rng: Random generator.

    Returns:
        Boolean frame over MASKED, True where the value is hidden.
    """
    shape = (len(X), len(MASKED))
    if mechanism == "mcar":
        p = np.full(shape, MASK_RATE)
    else:
        rank = X[ANCHOR].rank(pct=True).to_numpy()
        p = np.repeat((2 * MASK_RATE * rank)[:, None], len(MASKED), axis=1)
    return pd.DataFrame(rng.random(shape) < p, columns=MASKED)


def impute(X: pd.DataFrame, method: str) -> pd.DataFrame:
    """
    Fill the NaNs of a numeric frame with one of the compared methods.

    KNN distances need a common scale, so the KNN branch standardises with the
    observed means and deviations and undoes it afterwards.
    """
    match method:
        case "median":
            filled = SimpleImputer(strategy="median").fit_transform(X)
        case "knn":
            mean, std = X.mean(), X.std()
            scaled = KNNImputer(n_neighbors=5).fit_transform((X - mean) / std)
            filled = scaled * std.to_numpy() + mean.to_numpy()
        case "mice":
            mice = IterativeImputer(
                estimator=BayesianRidge(), max_iter=15, random_state=SEED
            )
            filled = mice.fit_transform(X)
        case _:
            raise ValueError(f"Unknown imputation method: {method}")
    return pd.DataFrame(filled, columns=X.columns)


def score(truth: pd.DataFrame, filled: pd.DataFrame, mask: pd.DataFrame) -> dict:
    """Per-feature error on the hidden cells, scaled by the feature's spread."""
    result = {}
    for column in MASKED:
        hidden = mask[column].to_numpy()
        error = filled[column].to_numpy()[hidden] - truth[column].to_numpy()[hidden]
        spread = truth[column].std()
        result[column] = {
            "nrmse": float(np.sqrt(np.mean(error**2)) / spread),
            "bias": float(error.mean() / spread),
            "std_ratio": float(filled[column].std() / spread),
        }
    return result


def numeric_experiment(X: pd.DataFrame) -> dict:
    """Hide, impute and score under both mechanisms."""
    truth = X[NUMERIC].astype(float).reset_index(drop=True)
    # Age is always observed and correlates with the counts: context for the
    # model-based imputers, never itself hidden or scored.
    truth["age"] = X["age"].map({a: i for i, a in enumerate(AGE_ORDER)}).to_numpy()
    rng = np.random.default_rng(SEED)

    results = {}
    for mechanism in ("mcar", "mar"):
        mask = make_mask(truth, mechanism, rng)
        observed = truth.copy()
        for column in MASKED:
            observed.loc[mask[column], column] = np.nan
        log.info(
            "%s: %.1f%% of the masked cells hidden",
            mechanism,
            mask.to_numpy().mean() * 100,
        )
        results[mechanism] = {
            m: score(truth, impute(observed, m), mask) for m in METHODS
        }
    return results


def plot_numeric(results: dict) -> None:
    """NRMSE and bias of each method under MCAR and MAR."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    x = np.arange(len(MASKED))
    width = 0.27
    titles = {"mcar": "MCAR", "mar": "MAR (пропуск залежить від time_in_hospital)"}
    for col, mechanism in enumerate(("mcar", "mar")):
        for row, (metric, ylabel) in enumerate(
            (("nrmse", "NRMSE"), ("bias", "Зміщення / σ"))
        ):
            ax = axes[row, col]
            for i, method in enumerate(METHODS):
                values = [results[mechanism][method][c][metric] for c in MASKED]
                ax.bar(
                    x + (i - 1) * width,
                    values,
                    width=width,
                    color=COLORS[method],
                    label=LABELS[method],
                )
            ax.set_ylabel(ylabel)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.grid(axis="y", alpha=0.3)
            if row == 0:
                ax.set_title(titles[mechanism])
            ax.set_xticks(x, MASKED, rotation=30, ha="right", fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(METHODS), fontsize=9)
    fig.suptitle(
        f"Відновлення прихованих {MASK_RATE:.0%} значень числових ознак".replace(
            "%", " %"
        )
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_figure(fig, LAB1 / "imputation_numeric.png")


def categorical_experiment(X: pd.DataFrame) -> dict:
    """Share of hidden categories each strategy recovers, plus the mode's distortion."""
    rng = np.random.default_rng(SEED)
    results = {}
    for column in CATEGORICAL_TARGETS:
        frame = X.reset_index(drop=True).copy()
        observed = frame[column].notna().to_numpy()
        hidden = observed & (rng.random(len(frame)) < MASK_RATE)
        truth = frame.loc[hidden, column].to_numpy()
        frame.loc[hidden, column] = np.nan

        scores = {}
        for strategy in ("mode", "knn"):
            imputer = CategoricalImputer(strategy=strategy, columns=(column,))
            filled = imputer.fit(frame).transform(frame)
            scores[strategy] = float(
                (filled.loc[hidden, column].to_numpy() == truth).mean()
            )

        # How far mode imputation of the real gaps would skew the distribution.
        full = X[column]
        mode = full.mode().iloc[0]
        results[column] = {
            "missing_pct": full.isna().mean() * 100,
            "n_levels": int(full.nunique()),
            "mode": mode,
            "mode_share_observed_pct": (full == mode).sum() / full.notna().sum() * 100,
            "mode_share_after_fill_pct": ((full == mode) | full.isna()).mean() * 100,
            "accuracy": scores,
        }
        log.info("%s: mode %.3f, knn %.3f", column, scores["mode"], scores["knn"])
    return results


def plot_categorical(results: dict) -> None:
    """Recovery accuracy per column, and how the mode would inflate its own share."""
    columns = list(results)
    x = np.arange(len(columns))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for i, (strategy, label, color) in enumerate(
        (("mode", "мода", "grey"), ("knn", "KNN-класифікатор (k = 15)", "steelblue"))
    ):
        values = [results[c]["accuracy"][strategy] * 100 for c in columns]
        bars = axes[0].bar(
            x + (i - 0.5) * 0.4, values, width=0.4, color=color, label=label
        )
        axes[0].bar_label(bars, fmt="%.0f", fontsize=7)
    axes[0].set_xticks(x, columns, rotation=30, ha="right", fontsize=8)
    axes[0].set_ylabel("Точність відновлення, %")
    axes[0].set_title("Відновлення прихованих категорій")
    axes[0].legend(fontsize=8)

    before = [results[c]["mode_share_observed_pct"] for c in columns]
    after = [results[c]["mode_share_after_fill_pct"] for c in columns]
    axes[1].bar(x - 0.2, before, width=0.4, color="grey", label="серед наявних значень")
    axes[1].bar(
        x + 0.2, after, width=0.4, color="crimson", label="після заповнення модою"
    )
    axes[1].set_xticks(x, columns, rotation=30, ha="right", fontsize=8)
    axes[1].set_ylabel("Частка найчастішої категорії, %")
    axes[1].set_title("Спотворення розподілу при заповненні модою")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, LAB1 / "imputation_categorical.png")


def run_imputation(X_train: pd.DataFrame, y_train: pd.Series) -> None:
    """Runs both experiments on a training subsample and writes imputation.json."""
    part = sample(X_train, y_train)
    numeric = numeric_experiment(part)
    plot_numeric(numeric)
    categorical = categorical_experiment(part)
    plot_categorical(categorical)
    save_json(
        {
            "sample_size": len(part),
            "mask_rate": MASK_RATE,
            "numeric": numeric,
            "categorical": categorical,
        },
        LAB1 / "imputation.json",
    )
