"""Hyperparameter study of the feature autoencoder, and the training of the chosen one.

The study varies one factor at a time, each step starting from the winner of
the step before: the backbone, then the patch size, then the two
regularisers, weight decay and masking noise. A full grid of the same values
would take 3 × 3 × 4 × 3 = 108 trainings; a step-wise search takes 10, which on
a CPU is the difference between an evening and a week. It assumes the factors
interact weakly, and the factors are ordered so that the one changing the
features themselves, the backbone, comes first.

Every choice is made on the validation part, by the best Pixel_AUROC any epoch
reached. The test part is not loaded here at all: lab 3 showed how much a
score the selection was made on overstates the model.

The model of the last step's winner is saved, with its training history, for
the benchmark.

Step names and figure labels stay in Ukrainian: they are rendered into the report.
"""

import gc
import logging
from dataclasses import asdict, replace

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from mk1.anomaly_metrics import ranking_metrics
from mk1.artifacts import save_figure, save_json
from mk1.config import LAB4
from mk1.feature_ae import EPOCHS, Config, FeatureAutoencoder, anomaly_maps, fit
from mk1.mvtec import Part, load, split
from mk1.patches import BACKBONES, PatchEmbedder, extract

log = logging.getLogger(__name__)

SEARCH_JSON = LAB4 / "search.json"
CHECKPOINT = LAB4 / "feature_ae.pt"

BASE = Config(backbone="resnet18")
PATCH_SIZES = (1, 3, 5)
WEIGHT_DECAYS = (0.0, 1e-2, 1e-1, 1.0)
DROPOUTS = (0.0, 0.2, 0.5)

STEP_LABELS = {
    "backbone": "мережа-екстрактор",
    "patch_size": "розмір патча p",
    "weight_decay": "weight decay",
    "dropout": "частка маскованих ознак",
}


class FeatureCache:
    """
    The training and validation features of one backbone at a time.

    Extraction is the slow part of a run, and every step after the first
    reuses one backbone, so its features are kept until another is asked for.

    Args:
        train: The training part.
        val: The validation part.
    """

    def __init__(self, train: Part, val: Part) -> None:
        self.train = train
        self.val = val
        self.backbone: str | None = None
        self.features: tuple[list[torch.Tensor], list[torch.Tensor]] = ([], [])
        self.channels = 0

    def get(self, backbone: str) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Training and validation features of the backbone, extracted if not cached."""
        if backbone != self.backbone:
            self.features = ([], [])
            gc.collect()
            log.info("Extracting %s features", backbone)
            embedder = PatchEmbedder(BACKBONES[backbone], patch_size=1)
            self.features = (
                extract(embedder, self.train.images),
                extract(embedder, self.val.images),
            )
            self.channels = embedder.channels
            self.backbone = backbone
        return self.features


def train_one(config: Config, cache: FeatureCache) -> tuple[dict, FeatureAutoencoder]:
    """
    Train one configuration and score its best epoch on the validation part.

    Args:
        config: The hyperparameters.
        cache: Features of the training and validation parts.

    Returns:
        The run's record and its model.
    """
    train_features, val_features = cache.get(config.backbone)
    model, history = fit(
        config, cache.channels, train_features, val_features, cache.val
    )
    maps = anomaly_maps(
        model, val_features, config.patch_size, tuple(cache.val.masks.shape[-2:])
    )
    best = max(history, key=lambda record: record["val_pixel_auroc"])
    record = {
        "config": asdict(config),
        "channels": cache.channels,
        "best_epoch": best["epoch"],
        "val": ranking_metrics(maps, cache.val),
        "seconds": history[-1]["seconds"],
        "history": history,
    }
    log.info(
        "%-58s val Pixel_AUROC %.4f (epoch %2d), PR-AUC %.4f, Image_AUROC %.4f, %.0fs",
        str(asdict(config)),
        record["val"]["pixel_auroc"],
        record["best_epoch"],
        record["val"]["pixel_pr_auc"],
        record["val"]["image_auroc"],
        record["seconds"],
    )
    return record, model


def run_search() -> None:
    """Runs the step-wise search; writes search.json, the chosen model and two figures."""
    splits = split()
    cache = FeatureCache(load(splits.train), load(splits.val))
    log.info(
        "Train %d images, validation %d (%d defective)",
        len(splits.train),
        len(splits.val),
        int(cache.val.anomalous.sum()),
    )

    done: dict[Config, tuple[dict, FeatureAutoencoder]] = {}
    steps: dict[str, list[Config]] = {}
    current = BASE
    candidates = {
        "backbone": lambda c: [replace(c, backbone=name) for name in BACKBONES],
        "patch_size": lambda c: [replace(c, patch_size=p) for p in PATCH_SIZES],
        "weight_decay": lambda c: [replace(c, weight_decay=w) for w in WEIGHT_DECAYS],
        "dropout": lambda c: [replace(c, dropout=d) for d in DROPOUTS],
    }
    for step, vary in candidates.items():
        log.info("--- step: %s ---", step)
        configs = vary(current)
        for config in configs:
            # The previous winner reappears in every step; it is not retrained.
            if config not in done:
                done[config] = train_one(config, cache)
        steps[step] = configs
        current = max(configs, key=lambda c: done[c][0]["val"]["pixel_auroc"])
        log.info("Step %s chose %s", step, asdict(current))

    record, model = done[current]
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": asdict(current),
            "channels": record["channels"],
            "state": model.state_dict(),
        },
        CHECKPOINT,
    )
    plot_search(steps, done)
    plot_training(record)
    save_json(
        {
            "epochs": EPOCHS,
            "selected": asdict(current),
            "selected_run": {
                key: value for key, value in record.items() if key != "history"
            },
            "trainings": len(done),
            "steps": {
                step: [done[config][0]["config"] for config in configs]
                for step, configs in steps.items()
            },
            "runs": [run for run, _ in done.values()],
        },
        SEARCH_JSON,
    )


def config_label(step: str, config: Config) -> str:
    """How the varied value of a step is printed on its axis."""
    match step:
        case "backbone":
            # Broken after the first hyphen: three long names share one narrow panel.
            label = BACKBONES[config.backbone].label
            return label.replace("-", "-\n", 1) if len(label) > 10 else label
        case "patch_size":
            return f"p = {config.patch_size}"
        case "weight_decay":
            return f"{config.weight_decay:g}"
        case _:
            return f"{config.dropout:g}"


def plot_search(
    steps: dict[str, list[Config]], done: dict[Config, tuple[dict, FeatureAutoencoder]]
) -> None:
    """
    Validation Pixel_AUROC and PR-AUC of every run, one column per step; the winner circled.

    Points rather than bars: the scores sit close together near the top of
    their range, and bars from zero would show them all as the same height.
    """
    metrics = (
        ("pixel_auroc", "Pixel_AUROC", "steelblue"),
        ("pixel_pr_auc", "Pixel PR-AUC", "darkorange"),
    )
    fig, axes = plt.subplots(2, len(steps), figsize=(16, 7.4), sharey="row")
    for column, (step, configs) in enumerate(steps.items()):
        x = np.arange(len(configs))
        winner = configs.index(
            max(configs, key=lambda c: done[c][0]["val"]["pixel_auroc"])
        )
        for row, (metric, title, colour) in enumerate(metrics):
            ax = axes[row, column]
            values = [done[c][0]["val"][metric] for c in configs]
            ax.plot(x, values, color=colour, marker="o", linewidth=1)
            ax.scatter(
                x[winner],
                values[winner],
                s=220,
                facecolors="none",
                edgecolors="black",
                linewidths=1.8,
                zorder=3,
            )
            for xi, value in zip(x, values):
                ax.annotate(
                    f"{value:.3f}",
                    (xi, value),
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )
            ax.set_xticks(
                x,
                [config_label(step, c) for c in configs] if row == 1 else [""] * len(x),
                fontsize=8,
            )
            ax.margins(x=0.2, y=0.3)
            ax.grid(alpha=0.3)
            if column == 0:
                ax.set_ylabel(f"{title}\n(валідаційна вибірка)")
        axes[0, column].set_title(STEP_LABELS[step], fontsize=11)
    save_figure(fig, LAB4 / "search.png")


def plot_training(record: dict) -> None:
    """Losses and validation Pixel_AUROC of the chosen configuration, per epoch."""
    history = record["history"]
    epochs = [r["epoch"] for r in history]
    fig, (loss_ax, auroc_ax) = plt.subplots(1, 2, figsize=(14, 4.6))
    loss_ax.plot(
        epochs,
        [r["train_loss"] for r in history],
        marker=".",
        label="навчальна вибірка",
    )
    loss_ax.plot(
        epochs,
        [r["val_loss"] for r in history],
        marker=".",
        label="валідаційна вибірка (зображення без дефектів)",
    )
    loss_ax.set_yscale("log")
    # Plain numbers at 1-2-5 steps rather than "4×10⁻¹".
    loss_ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    loss_ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    loss_ax.yaxis.set_minor_formatter(NullFormatter())
    loss_ax.set_xlabel("епоха")
    loss_ax.set_ylabel("MSE реконструкції (стандартизовані ознаки)")
    loss_ax.legend(fontsize=9)
    loss_ax.grid(alpha=0.3, which="both")

    auroc_ax.plot(
        epochs, [r["val_pixel_auroc"] for r in history], marker=".", color="steelblue"
    )
    auroc_ax.axvline(
        record["best_epoch"],
        color="grey",
        linestyle="--",
        label=f"найкраща епоха ({record['best_epoch']})",
    )
    auroc_ax.set_xlabel("епоха")
    auroc_ax.set_ylabel("Pixel_AUROC, валідаційна вибірка")
    auroc_ax.legend(fontsize=9)
    auroc_ax.grid(alpha=0.3)
    save_figure(fig, LAB4 / "training.png")
