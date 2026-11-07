"""The MVTec AD "bottle" category: locating, splitting and loading it, and its data report.

MVTec AD (Bergmann et al., 2019) is the reference benchmark of unsupervised
anomaly detection on industrial images. Its training part holds defect-free
images only; its test part mixes defect-free and defective ones, with a pixel
mask for every defect. "bottle" is its simplest category — one object, centred,
on a uniform background — and the targets are heavily imbalanced: a defect
covers a few percent of its image, so anomalous pixels are a small minority even
among the defective images.

The archive cannot be fetched by code: MVTec hands out the download link only
after a form that accepts the CC BY-NC-SA 4.0 licence. It is placed by hand at
ARCHIVE and extracted from there on first use.

There is no validation part in the benchmark, and choosing hyperparameters on
the test images would make the final score optimistic (the selection bias of
lab 3). So the test part is split in two, stratified by defect type: one half
validates every choice, the other is scored once at the end.

Defect names and figure labels stay in Ukrainian: they are rendered into the report.
"""

import logging
import tarfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from PIL import Image
from sklearn.model_selection import train_test_split

from mk1.artifacts import save_figure, save_json
from mk1.config import DATASETS, LAB4, SEED

log = logging.getLogger(__name__)

CATEGORY = "bottle"
ARCHIVE = DATASETS / f"{CATEGORY}.tar.xz"
EXTRACTED = DATASETS / "mvtec_ad"

# The input size of every model in the lab, PaDiM's and PatchCore's included,
# as in their papers: the backbones were pretrained near this scale.
IMAGE_SIZE = 256

# Normalisation the ImageNet-pretrained backbones were trained with.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Share of the benchmark's test part that becomes the validation part.
VALIDATION_SHARE = 0.5

GOOD = "good"
DEFECT_LABELS = {
    GOOD: "без дефекту",
    "broken_large": "великий скол",
    "broken_small": "малий скол",
    "contamination": "забруднення",
}

DATA_JSON = LAB4 / "data.json"


@dataclass(frozen=True)
class Sample:
    """
    One image of the category.

    Args:
        image: Path to the RGB image.
        mask: Path to the defect mask; None for a defect-free image.
        defect: The defect type, GOOD for a defect-free image.
    """

    image: Path
    mask: Path | None
    defect: str


@dataclass(frozen=True)
class Part:
    """
    A split loaded into memory at IMAGE_SIZE.

    Args:
        images: Normalised images, (N, 3, IMAGE_SIZE, IMAGE_SIZE) float32.
        masks: Defect masks, (N, IMAGE_SIZE, IMAGE_SIZE) bool; all False for a good image.
        defects: The defect type of each image.
    """

    images: torch.Tensor
    masks: torch.Tensor
    defects: list[str]

    @property
    def anomalous(self) -> np.ndarray:
        """Image-level labels: True for a defective image."""
        return np.array([defect != GOOD for defect in self.defects])


@dataclass(frozen=True)
class Splits:
    """The three parts every lab 4 stage works with."""

    train: list[Sample]
    val: list[Sample]
    test: list[Sample]


def locate() -> Path:
    """
    Return the category directory, extracting the archive on first use.

    Returns:
        Path to the directory holding train/, test/ and ground_truth/.

    Raises:
        FileNotFoundError: If neither the extracted category nor ARCHIVE exists.
    """
    category = EXTRACTED / CATEGORY
    if (category / "train").is_dir():
        log.debug("Dataset already extracted: %s", category)
        return category
    if not ARCHIVE.exists():
        raise FileNotFoundError(
            f"MVTec AD '{CATEGORY}' is missing. Download {ARCHIVE.name} from "
            "https://www.mvtec.com/company/research/datasets/mvtec-ad and place it "
            f"at {ARCHIVE}"
        )
    log.info("Extracting %s", ARCHIVE)
    EXTRACTED.mkdir(parents=True, exist_ok=True)
    with tarfile.open(ARCHIVE) as archive:
        # The "data" filter refuses absolute paths, links out of the target and
        # device files: the archive is a download, not a trusted file.
        archive.extractall(EXTRACTED, filter="data")
    if not (category / "train").is_dir():
        raise FileNotFoundError(f"{ARCHIVE.name} did not contain {CATEGORY}/train/")
    return category


def list_samples(category: Path) -> tuple[list[Sample], list[Sample]]:
    """
    List the training and test images of the category, with their masks.

    Args:
        category: The directory returned by locate().

    Returns:
        The training samples (all defect-free) and the test samples.

    Raises:
        FileNotFoundError: If a defective test image has no mask.
    """
    train = [
        Sample(path, None, GOOD)
        for path in sorted((category / "train" / GOOD).glob("*.png"))
    ]
    test = []
    for defect_dir in sorted((category / "test").iterdir()):
        for path in sorted(defect_dir.glob("*.png")):
            mask = None
            if defect_dir.name != GOOD:
                mask = (
                    category
                    / "ground_truth"
                    / defect_dir.name
                    / f"{path.stem}_mask.png"
                )
                if not mask.exists():
                    raise FileNotFoundError(f"No mask for {path}: expected {mask}")
            test.append(Sample(path, mask, defect_dir.name))
    return train, test


def split() -> Splits:
    """
    Cut the benchmark's test part into validation and test halves.

    Returns:
        The untouched training part and the two stratified halves.
    """
    train, test = list_samples(locate())
    val, held_out = train_test_split(
        test,
        test_size=1 - VALIDATION_SHARE,
        stratify=[sample.defect for sample in test],
        random_state=SEED,
    )
    return Splits(train=train, val=val, test=held_out)


def load_image(path: Path) -> torch.Tensor:
    """Read an image, resize it to IMAGE_SIZE and normalise it for the backbones."""
    with Image.open(path) as image:
        rgb = image.convert("RGB").resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
        )
    array = np.asarray(rgb, dtype=np.float32) / 255.0
    normalised = (array - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(normalised.astype(np.float32)).permute(2, 0, 1)


def load_mask(path: Path | None) -> torch.Tensor:
    """Read a defect mask at IMAGE_SIZE; an all-False mask for a defect-free image."""
    if path is None:
        return torch.zeros(IMAGE_SIZE, IMAGE_SIZE, dtype=torch.bool)
    with Image.open(path) as mask:
        # Nearest neighbour: an interpolated mask would invent partial labels.
        small = mask.convert("L").resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST
        )
    return torch.from_numpy(np.asarray(small) > 127)


def load(samples: list[Sample]) -> Part:
    """
    Load a split into memory.

    Args:
        samples: The images to load.

    Returns:
        The images, masks and defect types, in the order given.
    """
    return Part(
        images=torch.stack([load_image(sample.image) for sample in samples]),
        masks=torch.stack([load_mask(sample.mask) for sample in samples]),
        defects=[sample.defect for sample in samples],
    )


def to_display(image: torch.Tensor) -> np.ndarray:
    """Undo the normalisation of one image, for plotting."""
    array = image.permute(1, 2, 0).numpy() * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(array, 0.0, 1.0)


def run_data() -> None:
    """Describes the data and its split; writes data.json and three figures."""
    splits = split()
    with Image.open(splits.train[0].image) as first:
        original_size = first.size
    parts = {
        "train": splits.train,
        "val": splits.val,
        "test": splits.test,
    }
    counts = {
        name: dict(Counter(sample.defect for sample in samples))
        for name, samples in parts.items()
    }
    for name, count in counts.items():
        log.info("%-5s %d images: %s", name, sum(count.values()), count)

    loaded = {
        name: load(samples)
        for name, samples in (("val", splits.val), ("test", splits.test))
    }
    # Defect area as the share of its image, per defective image: the
    # imbalance a pixel-level detector faces.
    areas: dict[str, list[float]] = {}
    for part in loaded.values():
        for mask, defect in zip(part.masks, part.defects):
            if defect != GOOD:
                areas.setdefault(defect, []).append(mask.float().mean().item() * 100)
    anomalous_pixels = {
        name: part.masks.float().mean().item() * 100 for name, part in loaded.items()
    }
    all_areas = np.concatenate([np.array(values) for values in areas.values()])
    log.info(
        "Defect area: median %.2f%%, range %.2f-%.2f%% of the image; anomalous "
        "pixels: val %.2f%%, test %.2f%%",
        np.median(all_areas),
        all_areas.min(),
        all_areas.max(),
        anomalous_pixels["val"],
        anomalous_pixels["test"],
    )

    plot_samples(loaded["val"], loaded["test"])
    plot_patches(load([splits.train[0]]).images[0])
    plot_areas(areas)
    save_json(
        {
            "category": CATEGORY,
            "original_size": original_size,
            "image_size": IMAGE_SIZE,
            "counts": counts,
            "defect_area_pct": {
                defect: {
                    "median": float(np.median(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "n": len(values),
                }
                for defect, values in areas.items()
            },
            "defect_area_pct_all": {
                "median": float(np.median(all_areas)),
                "min": float(all_areas.min()),
                "max": float(all_areas.max()),
            },
            "anomalous_pixels_pct": anomalous_pixels,
        },
        DATA_JSON,
    )


def plot_samples(val: Part, test: Part) -> None:
    """One image per defect type, plain and with its mask outlined."""
    images = torch.cat([val.images, test.images])
    masks = torch.cat([val.masks, test.masks])
    defects = val.defects + test.defects
    fig, axes = plt.subplots(2, len(DEFECT_LABELS), figsize=(14, 7.2))
    for column, (defect, label) in enumerate(DEFECT_LABELS.items()):
        index = defects.index(defect)
        picture = to_display(images[index])
        axes[0, column].imshow(picture)
        axes[0, column].set_title(label)
        axes[1, column].imshow(picture)
        if defect != GOOD:
            axes[1, column].contour(
                masks[index].numpy(), levels=[0.5], colors="red", linewidths=1.2
            )
            share = masks[index].float().mean().item() * 100
            axes[1, column].set_title(f"маска: {share:.1f}% зображення", fontsize=10)
        else:
            axes[1, column].set_title("маска порожня", fontsize=10)
    for ax in axes.flat:
        ax.set_axis_off()
    save_figure(fig, LAB4 / "samples.png")


def plot_patches(image: torch.Tensor, stride: int = 8) -> None:
    """
    The patch grid the feature maps impose on an image.

    Args:
        image: One normalised image.
        stride: Pixels per feature-map cell at the first extracted layer.
    """
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.imshow(to_display(image))
    for position in range(0, IMAGE_SIZE + 1, stride):
        ax.axhline(position - 0.5, color="white", linewidth=0.3, alpha=0.6)
        ax.axvline(position - 0.5, color="white", linewidth=0.3, alpha=0.6)
    centre = (IMAGE_SIZE // 2 // stride) * stride
    for size, colour, label in (
        (1, "red", f"комірка {stride}×{stride} пікс. (p = 1)"),
        (3, "orange", f"окіл p = 3 ({3 * stride}×{3 * stride} пікс.)"),
        (5, "yellow", f"окіл p = 5 ({5 * stride}×{5 * stride} пікс.)"),
    ):
        corner = centre - (size // 2) * stride - 0.5
        ax.add_patch(
            Rectangle(
                (corner, corner),
                size * stride,
                size * stride,
                fill=False,
                edgecolor=colour,
                linewidth=2,
                label=label,
            )
        )
    ax.legend(loc="lower left", fontsize=9, framealpha=0.85)
    ax.set_axis_off()
    save_figure(fig, LAB4 / "patches.png")


def plot_areas(areas: dict[str, list[float]]) -> None:
    """Defect area as a share of the image, per defect type."""
    fig, ax = plt.subplots(figsize=(10, 4.2))
    defects = [defect for defect in DEFECT_LABELS if defect in areas]
    rng = np.random.default_rng(SEED)
    for row, defect in enumerate(defects):
        values = np.array(areas[defect])
        jitter = rng.uniform(-0.12, 0.12, len(values))
        ax.scatter(values, row + jitter, s=22, alpha=0.75)
        ax.plot(
            [np.median(values)] * 2, [row - 0.3, row + 0.3], color="black", linewidth=2
        )
    ax.set_yticks(range(len(defects)), [DEFECT_LABELS[d] for d in defects])
    ax.set_xscale("log")
    # Plain numbers at 1-2-5 steps rather than powers of ten.
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("площа дефекту, % площі зображення (логарифмічна шкала)")
    ax.grid(axis="x", alpha=0.3, which="both")
    save_figure(fig, LAB4 / "defect_area.png")
