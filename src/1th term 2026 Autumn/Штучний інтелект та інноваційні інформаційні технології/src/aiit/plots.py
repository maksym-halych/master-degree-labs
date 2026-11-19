"""Learning curves and result plots (task 10).

Titles, axis labels and legends stay in Ukrainian: they are report content,
read off the rendered figure, not code the reader of this module has to follow.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aiit.figures import save_figure

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)


def learning_curves(histories: dict[str, dict], outpath: Path, title: str) -> None:
    """Train/val loss curves for every model configuration on one figure."""
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
    save_figure(fig, outpath)
    plt.close(fig)


def overfit_gap(histories: dict[str, dict], outpath: Path, title: str) -> None:
    """The val_loss - train_loss difference: overfitting read off directly (task 12)."""
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
    save_figure(fig, outpath)
    plt.close(fig)


def predicted_vs_actual(y_true, y_pred, outpath: Path, title: str) -> None:
    """Predicted/actual scatter plot for the best model."""
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_true, y_pred, s=4, alpha=0.25, edgecolors="none")

    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], color="crimson", linewidth=1.2)

    ax.set_xlabel("Фактичне значення (×100 тис. \\$)")
    ax.set_ylabel("Прогнозоване значення (×100 тис. \\$)")
    ax.set_title(title)

    fig.tight_layout()
    save_figure(fig, outpath)
    plt.close(fig)


def framework_comparison(tf_results: dict, pt_results: dict, outpath: Path) -> None:
    """Bar chart of RMSE and training time: TensorFlow against PyTorch."""
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
    save_figure(fig, outpath)
    plt.close(fig)


def ga_convergence(
    generations: list[dict],
    naive_generations: list[dict],
    reference: float,
    min_diversity: float,
    outpath: Path,
) -> None:
    """Best and median fitness per generation, next to the population's diversity.

    The diversity panel also shows the naive GA, the one without defences
    against degeneration, from the same seed.
    """
    index = [g["index"] for g in generations]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    axes[0].plot(index, [g["best"] for g in generations], marker="o", label="Найкраща")
    axes[0].plot(index, [g["median"] for g in generations], marker="s", label="Медіана")
    axes[0].axhline(
        reference,
        color="black",
        linestyle="--",
        linewidth=1,
        label="ЛР2: baseline-relu",
    )
    axes[0].set_xlabel("Покоління")
    axes[0].set_ylabel("RMSE (валідаційна вибірка)")
    axes[0].set_title("Пристосованість популяції")
    axes[0].legend(fontsize=8)

    axes[1].plot(index, [g["diversity"] for g in generations], marker="o", label="ГА")
    axes[1].plot(
        [g["index"] for g in naive_generations],
        [g["diversity"] for g in naive_generations],
        marker="s",
        color="gray",
        label="ГА без захисту від виродження",
    )
    # Where immigrants came in, the offspring's own diversity is marked below the
    # restored one, so the collapse they answered stays visible.
    restored = [g for g in generations if g["immigrants"]]
    if restored:
        axes[1].scatter(
            [g["index"] for g in restored],
            [g["offspring_diversity"] for g in restored],
            marker="x",
            color="crimson",
            zorder=3,
            label="Нащадки до введення іммігрантів",
        )
    axes[1].axhline(
        min_diversity,
        color="crimson",
        linestyle=":",
        linewidth=1,
        label="Поріг виродження",
    )
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("Покоління")
    axes[1].set_ylabel("Середня частка відмінних генів")
    axes[1].set_title("Різноманітність популяції")
    axes[1].legend(fontsize=8)

    for ax in axes:
        ax.set_xticks(index)

    fig.tight_layout()
    save_figure(fig, outpath)
    plt.close(fig)


def search_comparison(
    curves: dict[str, list[float]], reference: float, outpath: Path
) -> None:
    """Best fitness found so far against the number of trainings, per search method."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for name, fitness in curves.items():
        x = np.arange(1, len(fitness) + 1)
        (line,) = ax.plot(
            x, np.minimum.accumulate(fitness), linewidth=2, label=f"{name}: найкраще"
        )
        ax.scatter(
            x, fitness, s=10, alpha=0.35, color=line.get_color(), edgecolors="none"
        )

    ax.axhline(
        reference,
        color="black",
        linestyle="--",
        linewidth=1,
        label="ЛР2: baseline-relu",
    )
    # Diverged and badly configured networks score far above the rest; showing
    # them would squash the part of the plot where the searches differ.
    best = min(min(f) for f in curves.values())
    ax.set_ylim(best - 0.02, reference + 0.25)
    ax.set_xlabel("Кількість навчених конфігурацій")
    ax.set_ylabel("RMSE (валідаційна вибірка)")
    ax.set_title("Генетичний алгоритм проти випадкового пошуку")
    ax.legend(fontsize=8)

    fig.tight_layout()
    save_figure(fig, outpath)
    plt.close(fig)


def _value_label(value: object) -> str:
    if isinstance(value, bool):
        return "так" if value else "ні"
    if isinstance(value, str):
        return value
    # Scientific notation without the zero padding of the exponent: 1e-5, not 1e-05.
    return f"{value:g}".replace("e-0", "e-")


def gene_evolution(
    populations: list[list[list[int]]], genes: list[tuple[str, tuple]], outpath: Path
) -> None:
    """Share of the population carrying each value of each gene, per generation."""
    n_generations = len(populations)
    # Constrained layout keeps the long value labels of one panel from running
    # into the panel to its left.
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True, layout="constrained")

    for position, (ax, (label, values)) in enumerate(
        zip(axes.flat, genes, strict=True)
    ):
        share = np.zeros((len(values), n_generations))
        for generation, population in enumerate(populations):
            for genome in population:
                share[genome[position], generation] += 1 / len(population)

        image = ax.imshow(
            share, aspect="auto", origin="lower", cmap="Blues", vmin=0, vmax=1
        )
        ax.set_yticks(range(len(values)))
        ax.set_yticklabels([_value_label(v) for v in values], fontsize=8)
        ax.set_xticks(range(0, n_generations, 2))
        ax.set_title(label, fontsize=10)
        ax.grid(False)

    for ax in axes[-1]:
        ax.set_xlabel("Покоління")
    fig.colorbar(image, ax=axes, shrink=0.8, label="Частка популяції")
    fig.suptitle("Еволюція значень гіперпараметрів у популяції")
    save_figure(fig, outpath)
    plt.close(fig)


def seed_study(
    runs: list[dict],
    labels: dict[str, str],
    reference: float,
    outpath: Path,
) -> None:
    """Search methods compared over several seeds: progress, and val/test RMSE of winners."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), width_ratios=(1.4, 1, 1))

    # Runs spend different numbers of trainings; the mean curve covers only the
    # stretch every run reached.
    for name, label in labels.items():
        length = min(len(r[name]["fitness"]) for r in runs)
        curves = np.array(
            [np.minimum.accumulate(r[name]["fitness"][:length]) for r in runs]
        )
        x = np.arange(1, length + 1)
        (line,) = axes[0].plot(x, curves.mean(axis=0), linewidth=2, label=label)
        axes[0].fill_between(
            x,
            curves.min(axis=0),
            curves.max(axis=0),
            color=line.get_color(),
            alpha=0.15,
        )
    axes[0].axhline(
        reference,
        color="black",
        linestyle="--",
        linewidth=1,
        label="ЛР2: baseline-relu",
    )
    best = min(min(r[name]["fitness"]) for r in runs for name in labels)
    axes[0].set_ylim(best - 0.01, reference + 0.05)
    axes[0].set_xlabel("Кількість навчених конфігурацій")
    axes[0].set_ylabel("Найкраща RMSE (валідаційна вибірка)")
    axes[0].set_title(f"Середнє за {len(runs)} запусками, смуга — мін./макс.")
    axes[0].legend(fontsize=8)

    names = list(labels)
    rng = np.random.default_rng(0)
    # No lab 2 reference line here: it lies far above every winner and would
    # squash the spread between the methods, which is what these panels show.
    for ax, key, title in (
        (axes[1], "best_val_rmse", "Валідаційна вибірка"),
        (axes[2], "test_rmse", "Тестова вибірка"),
    ):
        values = [[r[name][key] for r in runs] for name in names]
        ax.boxplot(values, widths=0.5, showfliers=False, medianprops={"color": "black"})
        for position, column in enumerate(values, start=1):
            # Horizontal jitter only, so overlapping seeds stay distinguishable.
            jitter = rng.uniform(-0.12, 0.12, len(column))
            ax.scatter(position + jitter, column, s=14, alpha=0.7, zorder=3)
        ax.set_xticks(range(1, len(names) + 1))
        ax.set_xticklabels(
            [labels[n] for n in names], rotation=15, ha="right", fontsize=8
        )
        ax.set_ylabel("RMSE найкращої конфігурації")
        ax.set_title(title)

    fig.tight_layout()
    save_figure(fig, outpath)
    plt.close(fig)
