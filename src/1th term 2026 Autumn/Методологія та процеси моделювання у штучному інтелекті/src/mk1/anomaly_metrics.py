"""Quality of an anomaly map, at the pixel and at the image level.

Every model of lab 4 — the proposed one and the Anomalib baselines — returns
the same thing, a score per pixel, and is scored here by the same code, so the
comparison cannot be skewed by differing metric implementations.

- Pixel_AUROC ranks every pixel of the split against the defect masks: the
  metric the assignment names and the one the MVTec AD papers report.
- Pixel PR-AUC (average precision) is the same ranking judged on the anomalous
  pixels only. They are a few percent of all pixels, and ROC-AUC is
  insensitive to how many normal pixels outrank them (the lesson of lab 1's
  PR-AUC, Saito & Rehmsmeier, 2015).
- Image_AUROC ranks the images by their highest pixel score.
- F1 needs a threshold, and a threshold is a choice: it is set where F1 peaks
  on the validation part and then applied unchanged to the test part, so the
  test F1 is that of a detector that has never seen the test images.
"""

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

from mk1.mvtec import Part


def image_scores(maps: torch.Tensor) -> np.ndarray:
    """An image's score: the highest score of any of its pixels."""
    return maps.flatten(1).amax(dim=1).numpy()


def best_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    The threshold at which F1 is highest on the given scores.

    Args:
        scores: Anomaly scores, higher meaning more anomalous.
        labels: True for an anomalous item.

    Returns:
        The score at and above which an item is called anomalous.
    """
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    # The last precision/recall pair has no threshold: it is the empty prediction.
    f1 = (
        2
        * precision[:-1]
        * recall[:-1]
        / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    )
    return float(thresholds[np.argmax(f1)])


def f1_at(scores: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    """F1 of the decision `score >= threshold`."""
    predicted = scores >= threshold
    true_positive = np.sum(predicted & labels)
    denominator = predicted.sum() + labels.sum()
    return float(2 * true_positive / denominator) if denominator else 0.0


def pixel_auroc(maps: torch.Tensor, part: Part) -> float:
    """Pixel_AUROC of a split's anomaly maps against its defect masks."""
    return float(roc_auc_score(part.masks.flatten().numpy(), maps.flatten().numpy()))


def ranking_metrics(maps: torch.Tensor, part: Part) -> dict[str, float]:
    """
    The threshold-free metrics of one split.

    Args:
        maps: Anomaly maps, (N, 1, H, W) or (N, H, W).
        part: The split the maps belong to.

    Returns:
        Pixel_AUROC, pixel PR-AUC and Image_AUROC.
    """
    maps = maps.reshape(part.masks.shape)
    pixels, labels = maps.flatten().numpy(), part.masks.flatten().numpy()
    return {
        "pixel_auroc": float(roc_auc_score(labels, pixels)),
        "pixel_pr_auc": float(average_precision_score(labels, pixels)),
        "image_auroc": float(roc_auc_score(part.anomalous, image_scores(maps))),
    }


def score(
    val_maps: torch.Tensor, val: Part, test_maps: torch.Tensor, test: Part
) -> dict:
    """
    Every metric of a model, with F1 thresholds fixed on the validation part.

    Args:
        val_maps: Anomaly maps of the validation images, (N, 1, H, W) or (N, H, W).
        val: The validation part.
        test_maps: Anomaly maps of the test images.
        test: The test part.

    Returns:
        The validation and test metrics, and the two thresholds used.
    """
    val_maps = val_maps.reshape(val.masks.shape)
    test_maps = test_maps.reshape(test.masks.shape)
    pixel_threshold = best_threshold(
        val_maps.flatten().numpy(), val.masks.flatten().numpy()
    )
    image_threshold = best_threshold(image_scores(val_maps), val.anomalous)
    result: dict = {"thresholds": {"pixel": pixel_threshold, "image": image_threshold}}
    for name, maps, part in (("val", val_maps, val), ("test", test_maps, test)):
        result[name] = {
            **ranking_metrics(maps, part),
            "pixel_f1": f1_at(
                maps.flatten().numpy(), part.masks.flatten().numpy(), pixel_threshold
            ),
            "image_f1": f1_at(image_scores(maps), part.anomalous, image_threshold),
        }
    return result
