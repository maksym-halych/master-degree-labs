"""Nested cross-validation of every search strategy, and the lab's summary.

The outer loop cuts the training part into five folds. On each outer training
part a strategy runs its whole search, inner 3-fold CV included, exactly as in
mk1.tuning; the chosen hyperparameters are then refitted on that outer training
part and scored once on the outer fold the search never saw. The mean of those
five scores estimates how well the strategy, not one lucky candidate, performs
on new patients. The best inner score it is compared with is biased upwards:
it is the maximum of many noisy estimates, chosen on the same folds that
measured it.

Figure text stays in Ukrainian: it is report content.
"""

import json
import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import cross_validate

from mk1.artifacts import save_figure, save_json
from mk1.config import CV_FOLDS, INNER_FOLDS, LAB3
from mk1.tuning import (
    BASELINE_JSON,
    METHODS,
    N_JOBS,
    SCORING,
    SELECT,
    TUNING_JSON,
    Method,
    baseline_params,
    build_tuned,
    inner_folds,
    outer_folds,
    search,
)

log = logging.getLogger(__name__)

NESTED_JSON = LAB3 / "nested.json"
SUMMARY_JSON = LAB3 / "summary.json"


def holdout_scores(
    params: dict[str, float],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> dict[str, float]:
    """
    Fit the pipeline with the given hyperparameters and score it on held-out rows.

    Args:
        params: Boosting hyperparameters; see build_tuned().
        X_train: Rows to fit on.
        y_train: Their labels.
        X_valid: Rows to score.
        y_valid: Their labels.

    Returns:
        PR-AUC and ROC-AUC on the held-out rows.
    """
    pipe = build_tuned(params).fit(X_train, y_train)
    proba = pipe.predict_proba(X_valid)[:, 1]
    return {
        "pr_auc": average_precision_score(y_valid, proba),
        "roc_auc": roc_auc_score(y_valid, proba),
    }


def nested_cv(method: Method, X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Nested cross-validation of one strategy.

    Args:
        method: The search strategy.
        X: Training part features.
        y: Training part labels.

    Returns:
        Per outer fold: the best inner score, the outer score of the chosen
        hyperparameters, the hyperparameters and the search time; and the
        mean and spread of the inner and outer scores over the folds.
    """
    folds = []
    for k, (train_idx, valid_idx) in enumerate(outer_folds().split(X, y)):
        X_in, y_in = X.iloc[train_idx], y.iloc[train_idx]
        result = search(method, X_in, y_in)
        outer = holdout_scores(
            result.best_params, X_in, y_in, X.iloc[valid_idx], y.iloc[valid_idx]
        )
        folds.append(
            {
                "inner": result.best_score,
                "outer": outer[SELECT],
                "outer_roc_auc": outer["roc_auc"],
                "params": result.best_params,
                "wall_seconds": result.wall_seconds,
                "compute_seconds": result.compute_seconds,
            }
        )
        log.info(
            "%-24s outer fold %d: inner PR-AUC %.4f  outer %.4f  %6.1fs  %s",
            method.label,
            k + 1,
            result.best_score,
            outer[SELECT],
            result.wall_seconds,
            result.best_params,
        )
    inner = np.array([f["inner"] for f in folds])
    outer = np.array([f["outer"] for f in folds])
    return {
        "folds": folds,
        "inner_mean": float(inner.mean()),
        "inner_std": float(inner.std(ddof=1)),
        "outer_mean": float(outer.mean()),
        "outer_std": float(outer.std(ddof=1)),
        "outer_roc_auc_mean": float(np.mean([f["outer_roc_auc"] for f in folds])),
        "optimism": float((inner - outer).mean()),
    }


def run_nested(X_train: pd.DataFrame, y_train: pd.Series) -> None:
    """Runs nested CV of every strategy; writes nested.json and a figure."""
    results: dict[str, dict] = {}
    for method in METHODS:
        started = time.perf_counter()
        result = nested_cv(method, X_train, y_train)
        result["wall_seconds"] = time.perf_counter() - started
        results[method.key] = result
        log.info(
            "%-24s nested: inner %.4f ± %.4f  outer %.4f ± %.4f  optimism %+.4f  %6.0fs",
            method.label,
            result["inner_mean"],
            result["inner_std"],
            result["outer_mean"],
            result["outer_std"],
            result["optimism"],
            result["wall_seconds"],
        )
    plot_nested(results)
    save_json(
        {
            "outer_folds": CV_FOLDS,
            "inner_folds": INNER_FOLDS,
            "methods": {m.key: m.label for m in METHODS},
            "results": results,
        },
        NESTED_JSON,
    )


def plot_nested(results: dict[str, dict]) -> None:
    """Inner against outer PR-AUC per strategy: fold means as bars, folds as dots."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(METHODS))
    width = 0.36
    for i, method in enumerate(METHODS):
        r = results[method.key]
        for offset, key, hatch in (
            (-width / 2, "inner", "//"),
            (width / 2, "outer", None),
        ):
            ax.bar(
                i + offset,
                r[f"{key}_mean"],
                width,
                yerr=r[f"{key}_std"],
                capsize=3,
                color=method.color,
                alpha=0.45 if key == "inner" else 0.9,
                hatch=hatch,
                edgecolor="black",
                linewidth=0.5,
            )
        # Each outer fold's pair of scores, joined: the slope is that fold's optimism.
        for f in r["folds"]:
            ax.plot(
                [i - width / 2, i + width / 2],
                [f["inner"], f["outer"]],
                color="black",
                alpha=0.35,
                lw=0.8,
                marker="o",
                markersize=3,
            )
    ax.set_xticks(x, [m.label for m in METHODS], fontsize=9)
    ax.set_ylabel("PR-AUC")
    low = min(min(f["outer"] for f in r["folds"]) for r in results.values())
    high = max(max(f["inner"] for f in r["folds"]) for r in results.values())
    ax.set_ylim(low - 0.01, high + 0.008)
    ax.grid(axis="y", alpha=0.3)
    handles = [
        plt.Rectangle(
            (0, 0), 1, 1, facecolor="lightgrey", hatch="//", edgecolor="black"
        ),
        plt.Rectangle((0, 0), 1, 1, facecolor="grey", edgecolor="black"),
        Line2D([], [], color="black", alpha=0.5, marker="o", markersize=3),
    ]
    ax.legend(
        handles,
        [
            "найкраща оцінка внутрішньої CV",
            "незміщена оцінка зовнішньої CV",
            "окремий зовнішній блок",
        ],
        loc="lower left",
        fontsize=9,
    )
    ax.set_title("Вкладена крос-валідація: 5 зовнішніх × 3 внутрішні блоки")
    save_figure(fig, LAB3 / "nested_folds.png")


def run_summary(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    """
    Score each strategy's choice on the test part and draw quality against time.

    Reads baseline.json, tuning.json and nested.json, so the baseline, tuning
    and nested stages must have run at least once. Each strategy's gain is
    measured against the untuned model on the same folds (baseline_nested),
    since the fold scheme alone moves the score. The test part is touched
    here once per strategy, after every choice has already been made, and
    chooses nothing. Writes summary.json and a figure.
    """
    baseline = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))
    tuning = json.loads(TUNING_JSON.read_text(encoding="utf-8"))["results"]
    nested = json.loads(NESTED_JSON.read_text(encoding="utf-8"))["results"]
    reference = baseline_nested(X_train, y_train)
    reference_outer = np.array([f["outer"] for f in reference["folds"]])

    rows: dict[str, dict] = {
        "baseline": {
            "candidates": 1,
            "wall_seconds": baseline["inner"]["wall_seconds"],
            "inner": baseline["inner"]["pr_auc"],
            "nested_inner": reference["inner_mean"],
            "nested": baseline["outer"]["pr_auc"],
            "nested_std": baseline["outer"]["pr_auc_std"],
            "test": holdout_scores(baseline_params(), X_train, y_train, X_test, y_test),
        }
    }
    for method in METHODS:
        t, n = tuning[method.key], nested[method.key]
        # Paired by outer fold: the folds differ in difficulty far more than the
        # strategies do, and pairing removes that shared part of the spread.
        gain = np.array([f["outer"] for f in n["folds"]]) - reference_outer
        rows[method.key] = {
            "candidates": len(t["history"]),
            "wall_seconds": t["wall_seconds"],
            "compute_seconds": t["compute_seconds"],
            "inner": t["best_score"],
            "inner_gain": t["best_score"] - baseline["inner"]["pr_auc"],
            "nested_inner": n["inner_mean"],
            "nested_inner_gain": n["inner_mean"] - reference["inner_mean"],
            "nested": n["outer_mean"],
            "nested_std": n["outer_std"],
            "nested_gain": float(gain.mean()),
            "nested_gain_std": float(gain.std(ddof=1)),
            "test": holdout_scores(t["best_params"], X_train, y_train, X_test, y_test),
        }
    for key, row in rows.items():
        log.info(
            "%-10s %3d candidates %7.1fs  inner %.4f  nested inner %.4f  "
            "nested %.4f ± %.4f  gain %+.4f  test PR-AUC %.4f  ROC-AUC %.4f",
            key,
            row["candidates"],
            row["wall_seconds"],
            row["inner"],
            row["nested_inner"],
            row["nested"],
            row["nested_std"],
            row.get("nested_gain", 0.0),
            row["test"]["pr_auc"],
            row["test"]["roc_auc"],
        )
    plot_quality_time(rows)
    save_json({"rows": rows, "baseline_nested": reference}, SUMMARY_JSON)


def baseline_nested(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Lab 2's untuned boosting, scored on the nested scheme's own folds.

    An inner score comes from a model fitted on two thirds of an outer training
    part, an outer score from one fitted on all of it, so the two differ even
    when nothing was selected. The untuned model on the same folds measures
    that difference; what a strategy's inner-outer gap holds beyond it is the
    optimism of selection.

    Args:
        X: Training part features.
        y: Training part labels.

    Returns:
        Per outer fold, the mean inner 3-fold PR-AUC and the outer PR-AUC; and
        the means of both over the folds.
    """
    folds = []
    for train_idx, valid_idx in outer_folds().split(X, y):
        X_in, y_in = X.iloc[train_idx], y.iloc[train_idx]
        inner = cross_validate(
            build_tuned({}),
            X_in,
            y_in,
            scoring=SCORING,
            cv=inner_folds(),
            n_jobs=N_JOBS,
        )
        outer = holdout_scores(
            baseline_params(), X_in, y_in, X.iloc[valid_idx], y.iloc[valid_idx]
        )
        folds.append(
            {"inner": float(np.mean(inner[f"test_{SELECT}"])), "outer": outer[SELECT]}
        )
    log.info(
        "Baseline on the nested folds: inner %.4f, outer %.4f",
        np.mean([f["inner"] for f in folds]),
        np.mean([f["outer"] for f in folds]),
    )
    return {
        "folds": folds,
        "inner_mean": float(np.mean([f["inner"] for f in folds])),
        "outer_mean": float(np.mean([f["outer"] for f in folds])),
    }


def plot_quality_time(rows: dict[str, dict]) -> None:
    """The three estimates of each strategy against the wall time of its search."""
    fig, ax = plt.subplots(figsize=(12, 6))
    styles = [("baseline", "без оптимізації (лаб. 2)", "black")] + [
        (m.key, m.label, m.color) for m in METHODS
    ]
    for key, label, color in styles:
        row = rows[key]
        t = row["wall_seconds"]
        ax.scatter(
            t, row["inner"], s=90, facecolors="none", edgecolors=color, linewidths=2
        )
        ax.errorbar(
            t,
            row["nested"],
            yerr=row["nested_std"],
            fmt="o",
            markersize=9,
            color=color,
            capsize=4,
            label=label,
        )
        ax.scatter(
            t, row["test"]["pr_auc"], marker="x", s=80, color=color, linewidths=2
        )
        ax.plot([t, t], [row["nested"], row["inner"]], color=color, lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("Час пошуку на всій навчальній вибірці, с (лог. шкала)")
    ax.set_ylabel("PR-AUC")
    ax.set_title("Якість моделі залежно від часу обчислень")
    ax.grid(alpha=0.3, which="both")
    # Both legends go outside the axes: the points of five strategies spread
    # over the whole plot, and any corner inside it would hide one of them.
    first = ax.legend(
        loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9, title="метод"
    )
    ax.add_artist(first)
    markers = [
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            markerfacecolor="none",
            markeredgecolor="black",
            markersize=9,
        ),
        Line2D([], [], marker="o", ls="", color="black", markersize=8),
        Line2D([], [], marker="x", ls="", color="black", markersize=8),
    ]
    ax.legend(
        markers,
        [
            "найкраща оцінка внутрішньої CV",
            "незміщена оцінка Nested CV (± σ)",
            "тестова вибірка",
        ],
        loc="lower left",
        bbox_to_anchor=(1.02, 0.0),
        fontsize=9,
        title="оцінка",
    )
    save_figure(fig, LAB3 / "quality_time.png")
