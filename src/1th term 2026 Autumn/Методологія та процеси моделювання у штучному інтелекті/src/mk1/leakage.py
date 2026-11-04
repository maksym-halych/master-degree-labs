"""How much a leaking pipeline inflates the cross-validated score.

Every variant feeds the same model, lab 2's untuned boosting, the same rows and
the same five folds; the variants differ only in which steps learn from the
whole training part before the folds are cut. Each one is also fitted on the
whole training part and scored once on the test part, which no variant ever
learns from: the test score is the honest one, and its distance from the
cross-validated score is the inflation the leak caused.

The incorrect pipeline the assignment asks for is "fit-transform": the
preprocessing is fitted on the whole training part, which is then transformed
with it, the routine scaler recipe. For target encoding that recipe leaks every
row's own label into its encoding; scikit-learn's TargetEncoder.fit_transform
cross-fits exactly to prevent it, which the "fit_transform" variant measures.

Variant labels stay in Ukrainian: they are the tick labels of a report figure.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import cross_validate

from mk1.artifacts import save_figure, save_json
from mk1.augmentation import make_sampler
from mk1.config import CV_FOLDS, LAB3
from mk1.tuning import N_JOBS, SCORING, build_tuned, outer_folds

log = logging.getLogger(__name__)

LEAKAGE_JSON = LAB3 / "leakage.json"


@dataclass(frozen=True)
class Matrices:
    """
    What a variant cross-validates and what it is tested on.

    Args:
        model: The unfitted estimator that the folds refit.
        X: Its training input: raw rows, or a matrix prepared before the split.
        y: Training labels, including any synthetic rows.
        X_test: The test part, prepared the way a new patient would be.
    """

    model: ClassifierMixin
    X: pd.DataFrame | np.ndarray
    y: pd.Series | np.ndarray
    X_test: pd.DataFrame | np.ndarray


def correct(X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame) -> Matrices:
    """The whole pipeline inside the folds: every step refitted on each training fold."""
    return Matrices(build_tuned({}), X, y, X_test)


def prefit_fit_transform(
    X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame
) -> Matrices:
    """Preprocessing learned on every training row via fit_transform (cross-fitted encoding)."""
    pipe = build_tuned({})
    head, model = pipe[:-1], pipe.named_steps["model"]
    Xt = head.fit_transform(X, y)
    return Matrices(model, Xt, y, head.transform(X_test))


def prefit_fit_then_transform(
    X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame
) -> Matrices:
    """Preprocessing fitted on every training row, which are then transformed with it."""
    pipe = build_tuned({})
    head, model = pipe[:-1], pipe.named_steps["model"]
    # fit, then transform: each row is target-encoded with means that include
    # its own label — the leak this variant exists to measure.
    head.fit(X, y)
    return Matrices(model, head.transform(X), y, head.transform(X_test))


def prefit_with_smote(X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame) -> Matrices:
    """The fit_transform variant, then SMOTE over-sampling of every training row."""
    pipe = build_tuned({})
    head, model = pipe[:-1], pipe.named_steps["model"]
    Xt = head.fit_transform(X, y)
    Xs, ys = make_sampler("smote").fit_resample(Xt, y)
    return Matrices(model, Xs, ys, head.transform(X_test))


@dataclass(frozen=True)
class Leak:
    """
    One way of preparing the data relative to the split.

    Args:
        key: Identifier used in the metric file.
        label: Name shown in the report figure.
        prepare: Builds the variant's matrices from the raw training and test parts.
    """

    key: str
    label: str
    prepare: Callable[[pd.DataFrame, pd.Series, pd.DataFrame], Matrices]


LEAKS = (
    Leak("correct", "коректний\nконвеєр", correct),
    Leak(
        "fit_transform",
        "препроцесинг\nдо розбиття\n(fit_transform)",
        prefit_fit_transform,
    ),
    Leak(
        "fit_then_transform",
        "препроцесинг\nдо розбиття\n(fit + transform)",
        prefit_fit_then_transform,
    ),
    Leak(
        "smote",
        "препроцесинг\nі SMOTE\nдо розбиття",
        prefit_with_smote,
    ),
)


def evaluate(
    leak: Leak, X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame, y_test: pd.Series
) -> dict:
    """
    Cross-validate one variant on the training part, then score it on the test part.

    Args:
        leak: The variant.
        X: Training part features, raw.
        y: Training part labels.
        X_test: Test part features, raw.
        y_test: Test part labels.

    Returns:
        Per-fold and mean validation metrics, test metrics, the inflation of
        each (validation mean minus test) and the size of the matrix the
        folds were cut from.
    """
    started = time.perf_counter()
    m = leak.prepare(X, y, X_test)
    cv = cross_validate(
        clone(m.model), m.X, m.y, scoring=SCORING, cv=outer_folds(), n_jobs=N_JOBS
    )
    final = clone(m.model).fit(m.X, m.y)
    proba = final.predict_proba(m.X_test)[:, 1]
    test = {
        "pr_auc": average_precision_score(y_test, proba),
        "roc_auc": roc_auc_score(y_test, proba),
    }
    out: dict = {
        "test": test,
        "rows": int(len(m.y)),
        "positive_pct": float(np.mean(m.y) * 100),
    }
    for metric in SCORING:
        folds = cv[f"test_{metric}"]
        out[metric] = {
            "folds": folds,
            "mean": float(np.mean(folds)),
            "std": float(np.std(folds, ddof=1)),
            "inflation": float(np.mean(folds) - test[metric]),
        }
    out["seconds"] = time.perf_counter() - started
    return out


def run_leakage(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    """Scores every variant on the folds and on the test part; writes leakage.json and a figure."""
    results: dict[str, dict] = {}
    for leak in LEAKS:
        result = evaluate(leak, X_train, y_train, X_test, y_test)
        results[leak.key] = result
        log.info(
            "%-20s CV PR-AUC %.4f ± %.4f  test %.4f  |  CV ROC-AUC %.4f  test %.4f  "
            "|  %d rows, %.1f%% positive, %.0fs",
            leak.key,
            result["pr_auc"]["mean"],
            result["pr_auc"]["std"],
            result["test"]["pr_auc"],
            result["roc_auc"]["mean"],
            result["test"]["roc_auc"],
            result["rows"],
            result["positive_pct"],
            result["seconds"],
        )
    correct_cv = results["correct"]["pr_auc"]["mean"]
    for key, result in results.items():
        log.info(
            "%-20s CV PR-AUC above the correct pipeline's by %+.4f",
            key,
            result["pr_auc"]["mean"] - correct_cv,
        )
    plot_leakage(results, y_train.mean())
    save_json(
        {
            "folds": CV_FOLDS,
            "variants": {leak.key: leak.label.replace("\n", " ") for leak in LEAKS},
            "results": results,
        },
        LEAKAGE_JSON,
    )


def plot_leakage(results: dict[str, dict], positive_rate: float) -> None:
    """Validation (mean ± fold σ) against test metrics, per variant."""
    panels = (("pr_auc", "PR-AUC"), ("roc_auc", "ROC-AUC"))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    x = np.arange(len(LEAKS))
    width = 0.38
    for ax, (metric, title) in zip(axes, panels):
        cv = [results[leak.key][metric]["mean"] for leak in LEAKS]
        std = [results[leak.key][metric]["std"] for leak in LEAKS]
        test = [results[leak.key]["test"][metric] for leak in LEAKS]
        ax.bar(
            x - width / 2,
            cv,
            width,
            yerr=std,
            capsize=3,
            color="indianred",
            label="валідаційні блоки (5-fold CV)",
        )
        ax.bar(x + width / 2, test, width, color="steelblue", label="тестова вибірка")
        # Each value sits above its bar's error bar, where nothing else is drawn.
        labelled = [
            (xi - width / 2, c, c + s) for xi, (c, s) in enumerate(zip(cv, std))
        ]
        labelled += [(xi + width / 2, t, t) for xi, t in enumerate(test)]
        for xpos, value, top in labelled:
            ax.annotate(
                f"{value:.3f}",
                (xpos, top),
                ha="center",
                va="bottom",
                fontsize=8,
                xytext=(0, 2),
                textcoords="offset points",
            )
        ax.set_xticks(x, [leak.label for leak in LEAKS], fontsize=9)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    axes[0].axhline(
        positive_rate, color="grey", linestyle="--", label="випадковий класифікатор"
    )
    axes[1].axhline(0.5, color="grey", linestyle="--")
    axes[1].set_ylim(0.45, None)
    axes[0].legend(loc="upper left", fontsize=8)
    save_figure(fig, LAB3 / "leakage.png")
