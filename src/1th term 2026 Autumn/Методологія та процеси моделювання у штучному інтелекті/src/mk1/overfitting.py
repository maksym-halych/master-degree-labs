"""Under which conditions over-sampling leads to overfitting: two controlled sweeps.

The capacity sweep varies how closely the random forest may fit its training
rows, with and without SMOTE. The ratio sweep varies how many synthetic rows
SMOTE adds, for every model. Both score the training rows and the validation
folds, so the overfitting shows as the distance between the two.

Figure titles, axis labels and legends stay in Ukrainian: they are report
content, read off the rendered figure.
"""

import logging
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline

from mk1.artifacts import save_figure, save_json
from mk1.comparison import MODEL_COLORS, MODEL_LABELS, cross_validate, log_result
from mk1.config import LAB2
from mk1.scenarios import MODELS, SCENARIO_BY_KEY, Scenario, build_scenario

log = logging.getLogger(__name__)

# min_samples_leaf of the random forest: 1 grows every tree until each leaf is
# pure, 100 keeps the trees shallow.
LEAF_SIZES = (1, 3, 10, 30, 100)

# Minority-to-majority ratio after SMOTE; "auto" in imbalanced-learn means 1.0.
# The data's own ratio is about 0.1, so 0.25 is the mildest over-sampling.
RATIOS = (0.25, 0.5, 0.75, 1.0)


def capacity_pipeline(scenario: Scenario, leaf: int) -> Pipeline:
    """The scenario's random forest pipeline with the given leaf size."""
    return build_scenario(scenario, "rf").set_params(model__min_samples_leaf=leaf)


def ratio_pipeline(model: str, ratio: float) -> Pipeline:
    """The SMOTE scenario's pipeline with the given minority-to-majority ratio."""
    return build_scenario(SCENARIO_BY_KEY["C-smote"], model).set_params(
        sampler__sampling_strategy=ratio
    )


def run_overfitting(X_train: pd.DataFrame, y_train: pd.Series) -> None:
    """Runs both sweeps and writes overfitting.json and its two figures."""
    capacity: dict[str, list[dict]] = {}
    for key in ("B", "C-smote"):
        capacity[key] = []
        for leaf in LEAF_SIZES:
            build = partial(capacity_pipeline, SCENARIO_BY_KEY[key], leaf)
            result = cross_validate(build, X_train, y_train)
            log_result(f"capacity {key} leaf={leaf}", result)
            capacity[key].append({"min_samples_leaf": leaf, **result})

    # The ratio sweep's starting point is the cleaned data without SMOTE, at the
    # data's own class ratio.
    natural = y_train.mean() / (1 - y_train.mean())
    ratios: dict[str, list[dict]] = {}
    for model in MODELS:
        build = partial(build_scenario, SCENARIO_BY_KEY["B"], model)
        result = cross_validate(build, X_train, y_train)
        log_result(f"ratio {model} natural={natural:.3f}", result)
        ratios[model] = [{"ratio": natural, **result}]
        for ratio in RATIOS:
            result = cross_validate(
                partial(ratio_pipeline, model, ratio), X_train, y_train
            )
            log_result(f"ratio {model} {ratio}", result)
            ratios[model].append({"ratio": ratio, **result})

    plot_capacity(capacity)
    plot_ratios(ratios)
    save_json(
        {"natural_ratio": natural, "capacity": capacity, "ratios": ratios},
        LAB2 / "overfitting.json",
    )


def plot_capacity(capacity: dict[str, list[dict]]) -> None:
    """Training and validation scores of the random forest against its leaf size."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    styles = {"B": ("black", "Б: без аугментації"), "C-smote": ("crimson", "В: SMOTE")}
    for ax, (metric, label) in zip(
        axes, (("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"))
    ):
        for key, rows in capacity.items():
            color, name = styles[key]
            leaves = [r["min_samples_leaf"] for r in rows]
            ax.plot(
                leaves,
                [r["train"]["mean"][metric] for r in rows],
                "--s",
                color=color,
                label=f"{name}, навчальні дані",
            )
            ax.errorbar(
                leaves,
                [r["valid"]["mean"][metric] for r in rows],
                yerr=[r["valid"]["std"][metric] for r in rows],
                fmt="-o",
                capsize=3,
                color=color,
                label=f"{name}, валідаційні блоки",
            )
        ax.set_xscale("log")
        ax.set_xticks(LEAF_SIZES, [str(v) for v in LEAF_SIZES])
        ax.set_xlabel("Мінімальна кількість записів у листку (min_samples_leaf)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Випадковий ліс: складність моделі та аугментація SMOTE")
    fig.tight_layout()
    save_figure(fig, LAB2 / "overfitting_capacity.png")


def plot_ratios(ratios: dict[str, list[dict]]) -> None:
    """Validation PR-AUC, train-validation gap and synthetic recall against the SMOTE ratio."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    ticks = np.round([ratios[MODELS[0]][0]["ratio"], *RATIOS], 2)
    for model, rows in ratios.items():
        x = [r["ratio"] for r in rows]
        color = MODEL_COLORS[model]
        axes[0].errorbar(
            x,
            [r["valid"]["mean"]["pr_auc"] for r in rows],
            yerr=[r["valid"]["std"]["pr_auc"] for r in rows],
            fmt="-o",
            capsize=3,
            color=color,
            label=MODEL_LABELS[model],
        )
        axes[1].plot(
            x,
            [
                r["train"]["mean"]["roc_auc"] - r["valid"]["mean"]["roc_auc"]
                for r in rows
            ],
            "-o",
            color=color,
            label=MODEL_LABELS[model],
        )
        # The first point has no synthetic rows, so it has no synthetic recall.
        axes[2].plot(
            x[1:],
            [r["synthetic_recall"] for r in rows[1:]],
            "-o",
            color=color,
            label=f"{MODEL_LABELS[model]}: синтетичні записи",
        )
        axes[2].plot(
            x,
            [r["valid"]["mean"]["recall"] for r in rows],
            ":^",
            color=color,
            label=f"{MODEL_LABELS[model]}: реальні пацієнти (валідація)",
        )
    titles = (
        ("PR-AUC на валідаційних блоках", "PR-AUC"),
        ("Розрив ROC-AUC: навчальні дані − валідація", "Різниця ROC-AUC"),
        ("Частка класу 1, розпізнана при порозі 0,5", "Повнота (recall)"),
    )
    for ax, (title, ylabel) in zip(axes, titles):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Співвідношення класів 1 : 0 після SMOTE")
        ax.set_ylabel(ylabel)
        ax.set_xticks(ticks)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    # The recall panel's lines run along its top and bottom edges; the empty
    # band between them is where its long legend fits.
    axes[2].legend(fontsize=7, loc="upper left", bbox_to_anchor=(0.0, 0.92))
    fig.tight_layout()
    save_figure(fig, LAB2 / "overfitting_ratio.png")
