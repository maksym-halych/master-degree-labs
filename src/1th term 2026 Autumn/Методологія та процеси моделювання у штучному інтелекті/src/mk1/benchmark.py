"""Final scores and cost of the proposed model against the Anomalib baselines.

Contenders:

- the feature autoencoder the search chose, as saved by mk1.ae_search;
- the same configuration on the other two backbones, trained here, which
  isolates what the backbone costs and buys;
- PaDiM (Defard et al., 2021) and PatchCore (Roth et al., 2022), the
  reference methods of the field, in Anomalib's own implementation and with
  its default settings. PatchCore runs on WideResNet-50-2, its paper's choice,
  and on ResNet-18, so that it can be compared with the autoencoder on the same
  backbone.

All of them see the same images at the same size and normalisation, and are
scored by the same code (mk1.anomaly_metrics): the F1 thresholds come from the
validation part, every metric reported is the test part's, and the test part is
used here for the first and only time.

PaDiM on WideResNet-50-2 is left out on purpose: with Anomalib's 550 features
it keeps a 550 × 550 covariance for each of the 64 × 64 grid cells, over 5 GB
of float32, more than this machine has to spare.

Model names and figure labels stay in Ukrainian where they are rendered into the report.
"""

import gc
import logging
import time
from dataclasses import asdict, dataclass, replace

import matplotlib.pyplot as plt
import numpy as np
import torch
from anomalib.models.image.padim.torch_model import PadimModel
from anomalib.models.image.patchcore.torch_model import PatchcoreModel
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from torch import nn

from mk1.ae_search import CHECKPOINT
from mk1.anomaly_metrics import score
from mk1.artifacts import save_figure, save_json
from mk1.config import LAB4, SEED
from mk1.feature_ae import Config, FeatureAEDetector, FeatureAutoencoder, fit
from mk1.mvtec import DEFECT_LABELS, GOOD, Part, load, split, to_display
from mk1.patches import BACKBONES, EXTRACT_BATCH, PatchEmbedder, extract
from mk1.profiling import THREADS, cpu_name, profile

log = logging.getLogger(__name__)

BENCHMARK_JSON = LAB4 / "benchmark.json"

# Anomalib's defaults for both methods.
PADIM_LAYERS = ["layer1", "layer2", "layer3"]
PATCHCORE_LAYERS = ["layer2", "layer3"]
CORESET_RATIO = 0.1
NEIGHBOURS = 9


@dataclass(frozen=True)
class Result:
    """
    One contender's maps and cost.

    Args:
        key: Identifier in the JSON.
        label: Name in figures and tables.
        family: "ours" or "baseline", for the figure's markers.
        backbone: timm name of its backbone.
        val_maps: Anomaly maps of the validation part.
        test_maps: Anomaly maps of the test part.
        cost: What mk1.profiling.profile() measured.
        fit_seconds: Wall-clock time to train or fit it.
    """

    key: str
    label: str
    family: str
    backbone: str
    val_maps: torch.Tensor
    test_maps: torch.Tensor
    cost: dict
    fit_seconds: float | None


class AnomalibDetector(nn.Module):
    """An Anomalib torch model reduced to the interface of FeatureAEDetector."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Images (N, 3, H, W) to anomaly maps (N, 1, H, W)."""
        with torch.no_grad():
            return self.model(images).anomaly_map


def backbone_of(detector: nn.Module) -> nn.Module:
    """
    The pretrained feature extractor inside a detector.

    Raises:
        TypeError: If the detector is neither of the two kinds benchmarked.
    """
    if isinstance(detector, FeatureAEDetector):
        return detector.embedder.extractor
    if isinstance(detector, AnomalibDetector):
        return detector.model.feature_extractor
    raise TypeError(f"Unknown detector type: {type(detector).__name__}")


def predict(detector: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Anomaly maps of a whole split, in batches."""
    detector.eval()
    return torch.cat(
        [
            detector(images[start : start + EXTRACT_BATCH])
            for start in range(0, len(images), EXTRACT_BATCH)
        ]
    )


def collect(model: nn.Module, images: torch.Tensor) -> None:
    """Feed the training images through an Anomalib model in training mode, where it stores them."""
    model.train()
    for start in range(0, len(images), EXTRACT_BATCH):
        model(images[start : start + EXTRACT_BATCH])


def fit_padim(train: Part) -> nn.Module:
    """PaDiM on ResNet-18: a Gaussian per grid cell over 100 random feature channels."""
    torch.manual_seed(SEED)  # the random channel subset
    model = PadimModel(backbone="resnet18", layers=PADIM_LAYERS, pre_trained=True)
    collect(model, train.images)
    model.fit()
    return AnomalibDetector(model)


def fit_patchcore(backbone: str, train: Part) -> nn.Module:
    """PatchCore: a coreset of the training patch embeddings, scored by nearest neighbour."""
    torch.manual_seed(SEED)  # the coreset's random projection and first centre
    model = PatchcoreModel(
        layers=PATCHCORE_LAYERS,
        backbone=backbone,
        pre_trained=True,
        num_neighbors=NEIGHBOURS,
    )
    collect(model, train.images)
    model.subsample_embedding(CORESET_RATIO)
    log.info("PatchCore %s memory bank: %s", backbone, tuple(model.memory_bank.shape))
    return AnomalibDetector(model)


def load_ours() -> tuple[Config, FeatureAEDetector]:
    """The autoencoder the search chose, as a full detector."""
    checkpoint = torch.load(CHECKPOINT, weights_only=True)
    config = Config(**checkpoint["config"])
    autoencoder = FeatureAutoencoder(checkpoint["channels"], dropout=config.dropout)
    autoencoder.load_state_dict(checkpoint["state"])
    embedder = PatchEmbedder(BACKBONES[config.backbone], config.patch_size)
    return config, FeatureAEDetector(embedder, autoencoder)


def train_ours(config: Config, train: Part, val: Part) -> FeatureAEDetector:
    """The chosen configuration on another backbone, trained as in the search."""
    embedder = PatchEmbedder(BACKBONES[config.backbone], config.patch_size)
    autoencoder, _ = fit(
        config,
        embedder.channels,
        extract(embedder, train.images),
        extract(embedder, val.images),
        val,
    )
    return FeatureAEDetector(embedder, autoencoder)


def run_benchmark() -> None:
    """Fits, scores and profiles every contender; writes benchmark.json and two figures."""
    splits = split()
    train, val, test = load(splits.train), load(splits.val), load(splits.test)
    example = test.images[:1]
    chosen, chosen_detector = load_ours()
    log.info(
        "Chosen configuration: %s; CPU: %s, %d threads", chosen, cpu_name(), THREADS
    )

    contenders = [
        ("ours", f"autoencoder-{name}", f"Автоенкодер ({backbone.label})", name)
        for name, backbone in BACKBONES.items()
    ]
    contenders += [
        ("baseline", "padim-resnet18", "PaDiM (ResNet-18)", "resnet18"),
        ("baseline", "patchcore-resnet18", "PatchCore (ResNet-18)", "resnet18"),
        (
            "baseline",
            "patchcore-wide_resnet50_2",
            "PatchCore (WideResNet-50-2)",
            "wide_resnet50_2",
        ),
    ]
    results: list[Result] = []
    for family, key, label, backbone in contenders:
        log.info("--- %s ---", key)
        started = time.time()
        if family == "ours" and backbone == chosen.backbone:
            # Trained by the search; its training time is in search.json.
            detector, fit_seconds = chosen_detector, None
        else:
            if family == "ours":
                detector = train_ours(replace(chosen, backbone=backbone), train, val)
            elif key.startswith("padim"):
                detector = fit_padim(train)
            else:
                detector = fit_patchcore(backbone, train)
            fit_seconds = time.time() - started
        results.append(
            Result(
                key=key,
                label=label,
                family=family,
                backbone=backbone,
                val_maps=predict(detector, val.images),
                test_maps=predict(detector, test.images),
                cost=profile(key, detector, backbone_of(detector), example),
                fit_seconds=fit_seconds,
            )
        )
        del detector
        gc.collect()

    scores = {r.key: score(r.val_maps, val, r.test_maps, test) for r in results}
    for r in results:
        metrics = scores[r.key]["test"]
        log.info(
            "%-28s test Pixel_AUROC %.4f  PR-AUC %.4f  F1 %.4f | Image_AUROC %.4f  F1 %.4f",
            r.key,
            metrics["pixel_auroc"],
            metrics["pixel_pr_auc"],
            metrics["pixel_f1"],
            metrics["image_auroc"],
            metrics["image_f1"],
        )
    plot_quality_cost(results, scores)
    plot_maps(results, test)
    save_json(
        {
            "chosen": asdict(chosen),
            "hardware": {"cpu": cpu_name(), "threads": THREADS, "batch_size": 1},
            "models": {
                r.key: {
                    "label": r.label,
                    "family": r.family,
                    "backbone": r.backbone,
                    "fit_seconds": r.fit_seconds,
                    **r.cost,
                    **scores[r.key],
                }
                for r in results
            },
        },
        BENCHMARK_JSON,
    )


def plot_quality_cost(results: list[Result], scores: dict) -> None:
    """Test Pixel_AUROC and PR-AUC against CPU latency, marker area by FLOPs."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4))
    panels = (("pixel_auroc", "Pixel_AUROC"), ("pixel_pr_auc", "Pixel PR-AUC"))
    for ax, (metric, title) in zip(axes, panels):
        for colour, r in zip(plt.cm.tab10.colors, results):
            ax.scatter(
                r.cost["latency"]["median_ms"],
                scores[r.key]["test"][metric],
                s=40 + 6 * r.cost["gflops"],
                # Circles for the proposed model, squares for the baselines.
                marker="o" if r.family == "ours" else "s",
                color=colour,
                alpha=0.8,
                edgecolor="black",
                label=f"{r.label}: {r.cost['gflops']:.1f} GFLOPs",
            )
        ax.set_xscale("log")
        # Plain numbers at 1-2-5 steps: a latency axis spans one or two
        # decades, and "2×10¹" is harder to read than "20".
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.margins(x=0.15, y=0.15)
        ax.set_xlabel(f"затримка на CPU, мс/зображення (batch = 1, {THREADS} потоків)")
        ax.set_ylabel(f"{title}, тестова вибірка")
        ax.grid(alpha=0.3, which="both")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=3,
        fontsize=9,
    )
    save_figure(fig, LAB4 / "quality_cost.png")


def plot_maps(results: list[Result], test: Part) -> None:
    """One test image per defect type: the image with its mask, then each model's map."""
    defects = [d for d in DEFECT_LABELS if d != GOOD]
    indices = [test.defects.index(d) for d in defects] + [test.defects.index(GOOD)]
    columns = len(results) + 1
    fig, axes = plt.subplots(
        len(indices), columns, figsize=(2.3 * columns, 2.4 * len(indices))
    )
    for row, index in enumerate(indices):
        picture = to_display(test.images[index])
        axes[row, 0].imshow(picture)
        axes[row, 0].contour(
            test.masks[index].numpy(), levels=[0.5], colors="red", linewidths=1
        )
        axes[row, 0].set_ylabel(DEFECT_LABELS[test.defects[index]], fontsize=9)
        for column, r in enumerate(results, start=1):
            maps = r.test_maps.reshape(test.masks.shape)
            # Each model's scores are on its own scale; the 1st-99.9th percentile
            # range of its test maps makes the colours comparable across rows.
            low, high = np.percentile(maps.numpy(), [1, 99.9])
            axes[row, column].imshow(picture)
            axes[row, column].imshow(
                maps[index].numpy(), cmap="jet", alpha=0.5, vmin=low, vmax=high
            )
    axes[0, 0].set_title("зображення і маска", fontsize=9)
    for column, r in enumerate(results, start=1):
        axes[0, column].set_title(r.label.replace(" (", "\n("), fontsize=8)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    save_figure(fig, LAB4 / "anomaly_maps.png")
