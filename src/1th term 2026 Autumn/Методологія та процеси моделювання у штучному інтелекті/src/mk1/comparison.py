"""Cross-validated and hold-out comparison of the classifiers across the data scenarios.

Every scenario-model pair is scored with stratified cross-validation on the
training part. The test part is scored once per pair, in run_test(), with the
decision threshold chosen on the training part's out-of-fold predictions.

Figure titles, axis labels and legends stay in Ukrainian: they are report
content, read off the rendered figure.
"""

import json
import logging
import time
from collections.abc import Callable
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from matplotlib.lines import Line2D
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold

from mk1.artifacts import save_figure, save_json
from mk1.config import CV_FOLDS, LAB2, SEED
from mk1.evaluation import best_threshold
from mk1.scenarios import MODELS, SCENARIOS, build_scenario, fit_on_matrix

log = logging.getLogger(__name__)

MODEL_LABELS = {
    "logreg": "логістична регресія",
    "rf": "випадковий ліс",
    "hgb": "градієнтний бустинг",
}
MODEL_COLORS = {"logreg": "steelblue", "rf": "forestgreen", "hgb": "darkorange"}
SCENARIO_COLORS = {
    "A": "grey",
    "B": "black",
    "C-smote": "crimson",
    "C-adasyn": "purple",
}

COMPARISON_JSON = LAB2 / "comparison.json"


def classification_scores(
    y_true: np.ndarray, proba: np.ndarray, threshold: float
) -> dict[str, float]:
    """
    Decision metrics at a threshold plus the threshold-free ones.

    Args:
        y_true: Binary labels.
        proba: Predicted probability of class 1.
        threshold: Probability at and above which a row is predicted class 1.

    Returns:
        Accuracy, precision, recall and F1 of class 1 at the threshold, plus
        ROC-AUC, PR-AUC and the Brier score of the probabilities.
    """
    predicted = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, predicted),
        "precision": precision_score(y_true, predicted, zero_division=0),
        "recall": recall_score(y_true, predicted, zero_division=0),
        "f1": f1_score(y_true, predicted, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "brier": brier_score_loss(y_true, proba),
    }


def summarise(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Mean and standard deviation of each metric over the folds."""
    table = pd.DataFrame(rows)
    return {"mean": table.mean().to_dict(), "std": table.std().to_dict()}


def cross_validate(
    build: Callable[[], Pipeline], X: pd.DataFrame, y: pd.Series
) -> dict:
    """
    Score one pipeline on the validation folds and on its own training rows.

    The training score is measured on the real rows of the training matrix
    only; the gap between it and the validation score is the measure of
    overfitting. The same training patients are also scored the way a new
    patient is, passed through the fitted pipeline: the matrix rows carry the
    cross-fitted target encoding, a new patient the full-fit one. For an
    over-sampling pipeline the share of synthetic rows the model classifies as
    class 1 is recorded too: a model that learned to recognise the synthetic
    points rather than the patients scores near 1 there and far lower on real
    ones.

    Args:
        build: Returns a fresh, unfitted pipeline.
        X: Training part features.
        y: Training part labels.

    Returns:
        Validation metrics at the threshold 0.5 and at the F1-optimal threshold
        of the out-of-fold predictions, training metrics as fitted and as new
        patients, that threshold, the mean fit time and the width and height
        of the training matrix.
    """
    folds = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    y_all = y.to_numpy()
    oof = np.zeros(len(y), dtype=float)
    valid_parts, train_rows, as_new, synthetic_recall, seconds = [], [], [], [], []
    converged = True
    for train_idx, valid_idx in folds.split(X, y):
        pipe = build()
        started = time.perf_counter()
        Xt, yt = fit_on_matrix(pipe, X.iloc[train_idx], y.iloc[train_idx])
        seconds.append(time.perf_counter() - started)
        model = pipe.named_steps["model"]
        if (
            isinstance(model, LogisticRegression)
            and model.n_iter_.max() >= model.max_iter
        ):
            converged = False

        oof[valid_idx] = pipe.predict_proba(X.iloc[valid_idx])[:, 1]
        n_real = len(train_idx)
        train_proba = model.predict_proba(Xt[:n_real])[:, 1]
        train_rows.append(classification_scores(yt[:n_real], train_proba, 0.5))
        new_proba = pipe.predict_proba(X.iloc[train_idx])[:, 1]
        as_new.append(classification_scores(yt[:n_real], new_proba, 0.5))
        if len(Xt) > n_real:
            synthetic = model.predict_proba(Xt[n_real:])[:, 1]
            synthetic_recall.append(float((synthetic >= 0.5).mean()))
        valid_parts.append(valid_idx)

    threshold, _ = best_threshold(y_all, oof)
    valid = [classification_scores(y_all[v], oof[v], 0.5) for v in valid_parts]
    tuned = [classification_scores(y_all[v], oof[v], threshold) for v in valid_parts]
    return {
        "valid": summarise(valid),
        "valid_tuned": summarise(tuned),
        "train": summarise(train_rows),
        "train_as_new": summarise(as_new),
        "synthetic_recall": float(np.mean(synthetic_recall))
        if synthetic_recall
        else None,
        "threshold": threshold,
        "fit_seconds": float(np.mean(seconds)),
        "n_features": int(Xt.shape[1]),
        "train_matrix_rows": int(len(Xt)),
        "converged": converged,
    }


def log_result(prefix: str, result: dict) -> None:
    """One console line per scored pipeline."""
    valid, train = result["valid"]["mean"], result["train"]["mean"]
    log.info(
        "%-28s PR-AUC %.4f ± %.4f  ROC-AUC %.4f  F1@0.5 %.3f  F1@%.3f %.3f  train ROC-AUC %.4f  %5.1fs",
        prefix,
        valid["pr_auc"],
        result["valid"]["std"]["pr_auc"],
        valid["roc_auc"],
        valid["f1"],
        result["threshold"],
        result["valid_tuned"]["mean"]["f1"],
        train["roc_auc"],
        result["fit_seconds"],
    )
    if not result["converged"]:
        log.warning("%s: logistic regression hit max_iter in at least one fold", prefix)


def run_comparison(X_train: pd.DataFrame, y_train: pd.Series) -> None:
    """Cross-validates every scenario with every model and writes comparison.json."""
    results: dict[str, dict[str, dict]] = {}
    for scenario in SCENARIOS:
        results[scenario.key] = {}
        for model in MODELS:
            result = cross_validate(
                partial(build_scenario, scenario, model), X_train, y_train
            )
            results[scenario.key][model] = result
            log_result(f"{scenario.key} {model}", result)
    plot_metrics(results, y_train.mean())
    plot_gap(results)
    save_json(
        {
            "folds": CV_FOLDS,
            "train_rows": len(X_train),
            "positive_pct": y_train.mean() * 100,
            "scenarios": {s.key: s.label for s in SCENARIOS},
            "results": results,
        },
        COMPARISON_JSON,
    )


def grouped_bars(ax: plt.Axes, results: dict, block: str, metric: str) -> None:
    """Bars of one metric: a group per scenario, a bar per model, ± one fold σ."""
    x = np.arange(len(SCENARIOS))
    width = 0.8 / len(MODELS)
    for i, model in enumerate(MODELS):
        means = [results[s.key][model][block]["mean"][metric] for s in SCENARIOS]
        stds = [results[s.key][model][block]["std"][metric] for s in SCENARIOS]
        ax.bar(
            x + (i - (len(MODELS) - 1) / 2) * width,
            means,
            width,
            yerr=stds,
            capsize=2,
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
        )
    ax.set_xticks(x, [s.short for s in SCENARIOS], fontsize=8)
    ax.grid(axis="y", alpha=0.3)


def plot_metrics(results: dict, positive_rate: float) -> None:
    """Every compared metric, per scenario and model, on the validation folds."""
    panels = (
        ("valid", "accuracy", "Accuracy (поріг 0,5)"),
        ("valid", "precision", "Precision (поріг 0,5)"),
        ("valid", "recall", "Recall (поріг 0,5)"),
        ("valid", "f1", "F1 (поріг 0,5)"),
        ("valid", "roc_auc", "ROC-AUC"),
        ("valid", "pr_auc", "PR-AUC"),
        ("valid_tuned", "f1", "F1 (оптимальний поріг)"),
        ("valid", "brier", "Оцінка Брієра (менше — краще)"),
    )
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.5))
    for ax, (block, metric, title) in zip(axes.ravel(), panels):
        grouped_bars(ax, results, block, metric)
        ax.set_title(title, fontsize=10)
    # Reference lines: what predicting "no readmission" for everyone scores,
    # and the PR-AUC of a classifier that ranks at random.
    always_zero = axes[0, 0].axhline(1 - positive_rate, color="grey", linestyle=":")
    axes[0, 0].set_ylim(0.5, 1.0)
    axes[1, 0].axhline(0.5, color="grey", linestyle="--")
    axes[1, 0].set_ylim(0.45, 0.7)
    random = axes[1, 1].axhline(positive_rate, color="grey", linestyle="--")
    bars, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        [*bars, always_zero, random],
        [*names, "завжди клас 0", "випадковий класифікатор"],
        loc="lower center",
        ncol=5,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save_figure(fig, LAB2 / "comparison_metrics.png")


def plot_gap(results: dict) -> None:
    """Training versus validation scores: the distance between them is the overfitting."""
    fig, axes = plt.subplots(2, len(MODELS), figsize=(15, 7.5), sharey="row")
    y_pos = np.arange(len(SCENARIOS))
    for row, (metric, label) in enumerate(
        (("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"))
    ):
        for ax, model in zip(axes[row], MODELS):
            train = [results[s.key][model]["train"]["mean"][metric] for s in SCENARIOS]
            valid = [results[s.key][model]["valid"]["mean"][metric] for s in SCENARIOS]
            ax.hlines(y_pos, valid, train, color="lightgrey", linewidth=3)
            ax.scatter(train, y_pos, color=MODEL_COLORS[model], marker="s", zorder=3)
            ax.scatter(
                valid,
                y_pos,
                edgecolors=MODEL_COLORS[model],
                facecolors="white",
                marker="o",
                zorder=3,
            )
            ax.set_yticks(y_pos, [s.label for s in SCENARIOS], fontsize=9)
            ax.invert_yaxis()
            ax.set_xlabel(label)
            ax.grid(axis="x", alpha=0.3)
            if row == 0:
                ax.set_title(MODEL_LABELS[model].capitalize(), fontsize=11)
    legend = (
        Line2D([], [], color="grey", marker="s", linestyle="", label="навчальні дані"),
        Line2D(
            [],
            [],
            color="grey",
            marker="o",
            markerfacecolor="white",
            linestyle="",
            label="валідаційні блоки",
        ),
    )
    fig.legend(handles=legend, loc="lower center", ncol=2, fontsize=9)
    fig.suptitle("Перенавчання: якість на навчальних даних і на валідаційних блоках")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_figure(fig, LAB2 / "overfitting_gap.png")


def load_thresholds() -> dict[tuple[str, str], float]:
    """
    The out-of-fold F1-optimal threshold of every scenario-model pair.

    Raises:
        FileNotFoundError: If the comparison stage has not been run yet.
    """
    if not COMPARISON_JSON.exists():
        raise FileNotFoundError(
            f"{COMPARISON_JSON} is missing: run the comparison stage before the test stage"
        )
    results = json.loads(COMPARISON_JSON.read_text(encoding="utf-8"))["results"]
    return {
        (scenario, model): result["threshold"]
        for scenario, by_model in results.items()
        for model, result in by_model.items()
    }


def run_test(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series
) -> None:
    """Fits every pair on the whole training part and scores it once on the test part."""
    thresholds = load_thresholds()
    y_true = y_test.to_numpy()
    results: dict[str, dict[str, dict]] = {}
    profiles: dict[str, dict] = {}
    curves: dict[tuple[str, str], np.ndarray] = {}
    for scenario in SCENARIOS:
        results[scenario.key] = {}
        for model in MODELS:
            pipe = build_scenario(scenario, model)
            Xt, yt = fit_on_matrix(pipe, X_train, y_train)
            # The data state does not depend on the model; record it once.
            profiles.setdefault(
                scenario.key,
                {
                    "rows": int(len(Xt)),
                    "synthetic_rows": int(len(Xt) - len(X_train)),
                    "positive_pct": float(yt.mean() * 100),
                    "features": int(Xt.shape[1]),
                },
            )
            proba = pipe.predict_proba(X_test)[:, 1]
            threshold = thresholds[(scenario.key, model)]
            predicted = (proba >= threshold).astype(int)
            results[scenario.key][model] = {
                "at_05": classification_scores(y_true, proba, 0.5),
                "tuned": classification_scores(y_true, proba, threshold),
                "threshold": threshold,
                "flagged": int(predicted.sum()),
                "true_positives": int((predicted & y_true).sum()),
            }
            curves[(scenario.key, model)] = proba
            tuned = results[scenario.key][model]["tuned"]
            log.info(
                "test %-10s %-6s PR-AUC %.4f ROC-AUC %.4f F1 %.3f at %.3f",
                scenario.key,
                model,
                tuned["pr_auc"],
                tuned["roc_auc"],
                tuned["f1"],
                threshold,
            )
    plot_curves(curves, y_true)
    save_json(
        {
            "test_rows": len(y_test),
            "positives": int(y_true.sum()),
            "profiles": profiles,
            "results": results,
        },
        LAB2 / "test.json",
    )


def plot_curves(curves: dict[tuple[str, str], np.ndarray], y_true: np.ndarray) -> None:
    """Precision-recall and ROC curves on the test part, a column per model."""
    fig, axes = plt.subplots(2, len(MODELS), figsize=(16, 9))
    for col, model in enumerate(MODELS):
        for scenario in SCENARIOS:
            key = scenario.key
            proba = curves[(key, model)]
            precision, recall, _ = precision_recall_curve(y_true, proba)
            axes[0, col].plot(
                recall,
                precision,
                color=SCENARIO_COLORS[key],
                linewidth=1.2,
                label=f"{scenario.label} ({average_precision_score(y_true, proba):.3f})",
            )
            fpr, tpr, _ = roc_curve(y_true, proba)
            axes[1, col].plot(
                fpr,
                tpr,
                color=SCENARIO_COLORS[key],
                linewidth=1.2,
                label=f"{scenario.label} ({roc_auc_score(y_true, proba):.3f})",
            )
        axes[0, col].axhline(y_true.mean(), color="grey", linestyle=":")
        axes[0, col].set_title(
            f"{MODEL_LABELS[model].capitalize()}: точність-повнота", fontsize=10
        )
        axes[0, col].set_xlabel("Повнота (recall)")
        axes[0, col].set_ylabel("Точність (precision)")
        axes[1, col].plot([0, 1], [0, 1], ":", color="grey")
        axes[1, col].set_title(
            f"{MODEL_LABELS[model].capitalize()}: ROC-крива", fontsize=10
        )
        axes[1, col].set_xlabel("Частка хибнопозитивних")
        axes[1, col].set_ylabel("Частка істиннопозитивних")
    for ax in axes.ravel():
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    save_figure(fig, LAB2 / "test_curves.png")
