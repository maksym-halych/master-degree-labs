"""Побудова навчальних кривих та графіків результатів (завдання 10)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)


def learning_curves(histories: dict[str, dict], outpath: Path, title: str) -> None:
    """Криві train/val loss для всіх конфігурацій моделі на одному рисунку."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for name, hist in histories.items():
        epochs = np.arange(1, len(hist["loss"]) + 1)
        axes[0].plot(epochs, hist["loss"], label=name)
        axes[1].plot(epochs, hist["val_loss"], label=name)

    axes[0].set_title("Тренувальна втрата (MSE)")
    axes[1].set_title("Валідаційна втрата (MSE)")
    for ax in axes:
        ax.set_xlabel("Епоха")
        ax.set_ylabel("MSE")
        ax.legend(fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def overfit_gap(histories: dict[str, dict], outpath: Path, title: str) -> None:
    """Різниця val_loss - train_loss: пряма діагностика перенавчання (завдання 12)."""
    fig, ax = plt.subplots(figsize=(7, 4))

    for name, hist in histories.items():
        epochs = np.arange(1, len(hist["loss"]) + 1)
        gap = np.array(hist["val_loss"]) - np.array(hist["loss"])
        ax.plot(epochs, gap, label=name)

    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Епоха")
    ax.set_ylabel("val_loss - train_loss")
    ax.set_title(title)
    ax.legend(fontsize=8)

    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def predicted_vs_actual(y_true, y_pred, outpath: Path, title: str) -> None:
    """Діаграма розсіювання прогноз/факт для найкращої моделі."""
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_true, y_pred, s=4, alpha=0.25, edgecolors="none")

    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], color="crimson", linewidth=1.2)

    ax.set_xlabel("Фактичне значення (×100 тис. \\$)")
    ax.set_ylabel("Прогнозоване значення (×100 тис. \\$)")
    ax.set_title(title)

    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def framework_comparison(tf_results: dict, pt_results: dict, outpath: Path) -> None:
    """Стовпчикова діаграма RMSE та часу навчання: TensorFlow проти PyTorch."""
    names = [k for k in tf_results if k in pt_results]
    x = np.arange(len(names))
    width = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].bar(
        x - width / 2,
        [tf_results[n]["test"]["rmse"] for n in names],
        width,
        label="TensorFlow",
    )
    axes[0].bar(
        x + width / 2,
        [pt_results[n]["test"]["rmse"] for n in names],
        width,
        label="PyTorch",
    )
    axes[0].set_ylabel("RMSE (тестова вибірка)")
    axes[0].set_title("Якість моделей")

    axes[1].bar(
        x - width / 2,
        [tf_results[n]["train_time_s"] for n in names],
        width,
        label="TensorFlow",
    )
    axes[1].bar(
        x + width / 2,
        [pt_results[n]["train_time_s"] for n in names],
        width,
        label="PyTorch",
    )
    axes[1].set_ylabel("Час навчання, с")
    axes[1].set_title("Швидкодія (20 епох, CPU)")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
        ax.legend()

    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
