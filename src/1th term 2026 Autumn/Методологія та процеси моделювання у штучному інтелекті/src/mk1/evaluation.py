"""Measured justification of every choice: cross-validated method comparisons.

Each experiment changes one field of the reference Variant and scores the
resulting pipeline with stratified cross-validation on the training part. The
test part is scored once, by the final configuration, in run_final().

Variant labels stay in Ukrainian: they are the tick labels of report figures.
"""

import logging
import time
from dataclasses import asdict, replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from mk1.artifacts import save_figure, save_json
from mk1.config import CV_FOLDS, LAB1, SEED
from mk1.pipeline import MODELS, Variant, build_pipeline

log = logging.getLogger(__name__)

REFERENCE = Variant()

EXPERIMENTS: dict[str, list[tuple[str, Variant]]] = {
    "imputation": [
        ("категорія Missing", REFERENCE),
        ("мода", replace(REFERENCE, impute="mode")),
        ("KNN", replace(REFERENCE, impute="knn")),
    ],
    "encoding": [
        ("Target", REFERENCE),
        ("Frequency", replace(REFERENCE, high_card="frequency")),
        ("One-Hot", replace(REFERENCE, high_card="onehot")),
    ],
    "scaling": [
        ("Yeo-Johnson", REFERENCE),
        ("Box-Cox", replace(REFERENCE, scaler="box-cox")),
        ("Robust", replace(REFERENCE, scaler="robust")),
        ("Standard", replace(REFERENCE, scaler="standard")),
    ],
    "outliers": [
        ("залишити", REFERENCE),
        ("вилучити (IF)", replace(REFERENCE, outliers="iforest")),
        ("вилучити (IQR)", replace(REFERENCE, outliers="iqr")),
    ],
    "features": [
        ("лише вихідні", replace(REFERENCE, domain=False, aggregates=False)),
        ("+ доменні", replace(REFERENCE, aggregates=False)),
        ("+ агрегати", REFERENCE),
        ("+ поліноміальні", replace(REFERENCE, polynomial=True)),
    ],
    "resampling": [
        ("без змін", REFERENCE),
        ("ваги класів", replace(REFERENCE, sampler="class-weight")),
        ("ROS", replace(REFERENCE, sampler="ros")),
        ("SMOTE", replace(REFERENCE, sampler="smote")),
        ("SMOTENC", replace(REFERENCE, sampler="smotenc")),
        # Isolates the interaction of SMOTENC with target encoding: the same
        # sampler, with the high-cardinality columns one-hot encoded instead.
        (
            "SMOTENC + One-Hot",
            replace(REFERENCE, sampler="smotenc", high_card="onehot"),
        ),
        ("ADASYN", replace(REFERENCE, sampler="adasyn")),
        ("гаусів шум", replace(REFERENCE, sampler="noise")),
        ("SMOTE-ENN", replace(REFERENCE, sampler="smote-enn")),
    ],
    # The "controlled" in controlled Gaussian noise: its scale, relative to
    # each feature's spread, chosen by cross-validation like any other setting.
    "noise": [
        (f"σ = {sigma}", replace(REFERENCE, sampler="noise", noise_sigma=sigma))
        for sigma in (0.05, 0.1, 0.2, 0.5, 1.0)
    ],
}

TITLES = {
    "imputation": "Імп'ютація категоріальних ознак",
    "encoding": "Кодування високопотужних ознак",
    "scaling": "Перетворення числових ознак",
    "outliers": "Обробка викидів",
    "features": "Інженерія ознак",
    "resampling": "Балансування класів",
    "noise": "Масштаб гаусового шуму",
}
MODEL_LABELS = {"logreg": "логістична регресія", "hgb": "градієнтний бустинг"}
MODEL_COLORS = {"logreg": "steelblue", "hgb": "darkorange"}

# Configurations chosen from the cross-validation results (experiments.json).
# The two models disagree on encoding and scaling, so each gets its own: a
# linear model needs what a tree ensemble is indifferent to, and vice versa.
FINALS = {
    "logreg": replace(REFERENCE, scaler="robust"),
    "hgb": replace(REFERENCE, high_card="onehot", aggregates=False),
}
# What a quick first attempt would use, for contrast on the test set.
NAIVE = Variant(
    impute="mode", high_card="onehot", scaler="standard", domain=False, aggregates=False
)

# One-factor-at-a-time comparisons miss interactions, so the combined
# configurations are cross-validated again before the test set is touched.
CANDIDATES = [
    ("еталонна", REFERENCE),
    ("підсумкова (логістична регресія)", FINALS["logreg"]),
    ("підсумкова (бустинг)", FINALS["hgb"]),
    ("наївна", NAIVE),
]


def best_threshold(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, float]:
    """The probability threshold with the highest F1, and that F1."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    # The last precision/recall pair has no threshold of its own.
    best = int(np.argmax(f1[:-1]))
    return float(thresholds[best]), float(f1[best])


def scores(
    y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """
    Threshold-free ranking metrics plus decisions at the given threshold.

    Accuracy is left out on purpose: predicting "no readmission" for everyone
    already scores 91%. The best achievable F1 is reported alongside F1 at the
    threshold: resampling moves predicted probabilities up, which changes F1 at
    a fixed threshold without changing what the model can rank.
    """
    predicted = (proba >= threshold).astype(int)
    return {
        "pr_auc": average_precision_score(y_true, proba),
        "roc_auc": roc_auc_score(y_true, proba),
        "f1": f1_score(y_true, predicted, zero_division=0),
        "f1_best": best_threshold(y_true, proba)[1],
        "precision": precision_score(y_true, predicted, zero_division=0),
        "recall": recall_score(y_true, predicted),
        "brier": brier_score_loss(y_true, proba),
    }


def cross_validate(variant: Variant, model: str, X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Score one pipeline with stratified K-fold cross-validation.

    Returns:
        Mean and standard deviation of every metric over the folds, the mean
        fit time, and the width of the matrix the model was trained on.
    """
    folds = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    per_fold, seconds = [], []
    for train_idx, valid_idx in folds.split(X, y):
        pipe = build_pipeline(variant, model)
        started = time.perf_counter()
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        seconds.append(time.perf_counter() - started)
        proba = pipe.predict_proba(X.iloc[valid_idx])[:, 1]
        per_fold.append(scores(y.iloc[valid_idx].to_numpy(), proba))
    table = pd.DataFrame(per_fold)
    n_features = pipe.named_steps["model"].n_features_in_
    return {
        "mean": table.mean().to_dict(),
        "std": table.std().to_dict(),
        "fit_seconds": float(np.mean(seconds)),
        "n_features": int(n_features),
    }


def run_experiments(X_train: pd.DataFrame, y_train: pd.Series) -> None:
    """Runs every experiment with both models and writes experiments.json."""
    # Variants shared between experiments (the reference above all) are
    # scored once and reused.
    cache: dict[tuple[Variant, str], dict] = {}
    results: dict[str, list[dict]] = {}
    for experiment, variants in EXPERIMENTS.items():
        results[experiment] = []
        for label, variant in variants:
            row = {"label": label, "variant": asdict(variant)}
            for model in MODELS:
                if (variant, model) not in cache:
                    cache[(variant, model)] = cross_validate(
                        variant, model, X_train, y_train
                    )
                result = cache[(variant, model)]
                row[model] = result
                log.info(
                    "%-10s %-18s %-6s PR-AUC %.4f ± %.4f  ROC-AUC %.4f  F1 %.3f  best F1 %.3f  %5.1fs",
                    experiment,
                    label,
                    model,
                    result["mean"]["pr_auc"],
                    result["std"]["pr_auc"],
                    result["mean"]["roc_auc"],
                    result["mean"]["f1"],
                    result["mean"]["f1_best"],
                    result["fit_seconds"],
                )
            results[experiment].append(row)
    plot_experiments(results)
    plot_noise(results["noise"])
    save_json(
        {
            "folds": CV_FOLDS,
            "train_rows": len(X_train),
            "positive_pct": y_train.mean() * 100,
            "results": results,
        },
        LAB1 / "experiments.json",
    )


def plot_experiments(results: dict) -> None:
    """Mean PR-AUC with its fold-to-fold deviation, per experiment and model."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    shown = [e for e in results if e != "noise"]
    for ax, experiment in zip(axes.ravel(), shown):
        rows = results[experiment]
        y_pos = np.arange(len(rows))
        for offset, model in zip((-0.15, 0.15), MODELS):
            means = [r[model]["mean"]["pr_auc"] for r in rows]
            stds = [r[model]["std"]["pr_auc"] for r in rows]
            ax.errorbar(
                means,
                y_pos + offset,
                xerr=stds,
                fmt="o",
                color=MODEL_COLORS[model],
                capsize=3,
                label=MODEL_LABELS[model],
            )
        ax.set_yticks(y_pos, [r["label"] for r in rows], fontsize=9)
        ax.invert_yaxis()
        ax.set_title(TITLES[experiment], fontsize=11)
        ax.set_xlabel("PR-AUC (середнє ± σ за 5 фолдами)", fontsize=8)
        ax.grid(axis="x", alpha=0.3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(MODELS), fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save_figure(fig, LAB1 / "experiments.png")


def plot_noise(rows: list[dict]) -> None:
    """Ranking quality and decisions at 0.5 as the noise scale grows."""
    sigmas = [r["variant"]["noise_sigma"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric, label in (
        (axes[0], "pr_auc", "PR-AUC"),
        (axes[1], "f1", "F1 (поріг 0,5)"),
    ):
        for model in MODELS:
            means = [r[model]["mean"][metric] for r in rows]
            stds = [r[model]["std"][metric] for r in rows]
            ax.errorbar(
                sigmas,
                means,
                yerr=stds,
                marker="o",
                capsize=3,
                color=MODEL_COLORS[model],
                label=MODEL_LABELS[model],
            )
        ax.set_xscale("log")
        ax.set_xlabel("σ шуму (у частках стандартного відхилення ознаки)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Керований гаусів шум: вибір масштабу крос-валідацією")
    fig.tight_layout()
    save_figure(fig, LAB1 / "noise_sweep.png")


def tuned_threshold(
    variant: Variant, model: str, X: pd.DataFrame, y: pd.Series
) -> float:
    """
    Decision threshold maximising F1 on out-of-fold training predictions.

    The threshold is a parameter like any other, so it is chosen on training
    data only; the test set then shows how well that choice carries over.
    """
    folds = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    oof = cross_val_predict(
        build_pipeline(variant, model), X, y, cv=folds, method="predict_proba"
    )[:, 1]
    return best_threshold(y.to_numpy(), oof)[0]


def run_final(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series
) -> None:
    """Fits the final and the naive pipelines on the whole training part and scores them once on the test part."""
    candidates = []
    for label, variant in CANDIDATES:
        row = {"label": label, "variant": asdict(variant)}
        for model in MODELS:
            row[model] = cross_validate(variant, model, X_train, y_train)
            log.info(
                "cv %-34s %-6s PR-AUC %.4f", label, model, row[model]["mean"]["pr_auc"]
            )
        candidates.append(row)

    results, curves = {}, {}
    y_true = y_test.to_numpy()
    for name in ("final", "naive"):
        for model in MODELS:
            variant = FINALS[model] if name == "final" else NAIVE
            threshold = tuned_threshold(variant, model, X_train, y_train)
            proba = (
                build_pipeline(variant, model)
                .fit(X_train, y_train)
                .predict_proba(X_test)[:, 1]
            )
            predicted = (proba >= threshold).astype(int)
            key = f"{name}-{model}"
            results[key] = {
                **scores(y_true, proba, threshold),
                "threshold": threshold,
                "flagged": int(predicted.sum()),
                "true_positives": int((predicted & y_true).sum()),
                "positives": int(y_true.sum()),
            }
            curves[key] = proba
            log.info(
                "test %-13s PR-AUC %.4f ROC-AUC %.4f F1 %.3f at %.3f",
                key,
                results[key]["pr_auc"],
                results[key]["roc_auc"],
                results[key]["f1"],
                threshold,
            )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    labels = {"final": "підсумковий конвеєр", "naive": "наївна обробка"}
    for key, proba in curves.items():
        name, model = key.split("-")
        style = "-" if name == "final" else "--"
        label = f"{labels[name]}, {MODEL_LABELS[model]}"
        precision, recall, _ = precision_recall_curve(y_true, proba)
        axes[0].plot(
            recall,
            precision,
            style,
            color=MODEL_COLORS[model],
            label=f"{label} ({results[key]['pr_auc']:.3f})",
        )
        fpr, tpr, _ = roc_curve(y_true, proba)
        axes[1].plot(
            fpr,
            tpr,
            style,
            color=MODEL_COLORS[model],
            label=f"{label} ({results[key]['roc_auc']:.3f})",
        )
    axes[0].axhline(
        y_true.mean(),
        color="grey",
        linestyle=":",
        label=f"випадковий класифікатор ({y_true.mean():.3f})",
    )
    axes[0].set_xlabel("Повнота (recall)")
    axes[0].set_ylabel("Точність (precision)")
    axes[0].set_title("Крива точність-повнота на тестовій вибірці")
    axes[1].plot([0, 1], [0, 1], ":", color="grey")
    axes[1].set_xlabel("Частка хибнопозитивних")
    axes[1].set_ylabel("Частка істиннопозитивних")
    axes[1].set_title("ROC-крива на тестовій вибірці")
    for ax in axes:
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    save_figure(fig, LAB1 / "final_curves.png")
    save_json(
        {
            "final_variants": {m: asdict(v) for m, v in FINALS.items()},
            "naive_variant": asdict(NAIVE),
            "cv_candidates": candidates,
            "test_rows": len(y_test),
            "results": results,
        },
        LAB1 / "final.json",
    )
