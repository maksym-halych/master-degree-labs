"""Outlier detection: IQR, Z-score, Isolation Forest and LOF compared (task 2).

Figure titles and axis labels stay in Ukrainian: they are report content, read
off the rendered figure, not code the reader of this module has to follow.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from mk1.artifacts import save_figure, save_json
from mk1.config import LAB1, SEED
from mk1.data import NUMERIC, TARGET

log = logging.getLogger(__name__)

METHODS = ("iqr", "zscore", "iforest", "lof")
LABELS = {
    "iqr": "IQR (1,5·IQR)",
    "zscore": "Z-score (|z| > 3)",
    "iforest": "Isolation Forest",
    "lof": "LOF",
}


def iqr_flags(X: pd.DataFrame, k: float = 1.5) -> pd.DataFrame:
    """Per-feature flags outside [Q1 - k·IQR, Q3 + k·IQR]."""
    q1, q3 = X.quantile(0.25), X.quantile(0.75)
    iqr = q3 - q1
    return (X < q1 - k * iqr) | (X > q3 + k * iqr)


def zscore_flags(X: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    """Per-feature flags more than `threshold` standard deviations from the mean."""
    return ((X - X.mean()) / X.std()).abs() > threshold


def outlier_mask(X: pd.DataFrame, method: str) -> np.ndarray:
    """
    Flag outlying rows with one of the compared methods.

    Args:
        X: The numeric features, one row per observation.
        method: One of METHODS.

    Returns:
        Boolean mask, True for an outlier.

    Raises:
        ValueError: If the method is unknown.
    """
    match method:
        case "iqr":
            return iqr_flags(X).any(axis=1).to_numpy()
        case "zscore":
            return zscore_flags(X).any(axis=1).to_numpy()
        case "iforest":
            forest = IsolationForest(
                n_estimators=200, contamination="auto", random_state=SEED
            )
            return forest.fit_predict(X) == -1
        case "lof":
            # LOF measures distance, so the features need a common scale. A
            # large neighbourhood is required because the counts repeat: with
            # 11% of rows sharing their exact numeric profile with another,
            # a small k would see neighbourhoods of zero radius.
            Z = StandardScaler().fit_transform(X)
            return (
                LocalOutlierFactor(n_neighbors=50, contamination="auto").fit_predict(Z)
                == -1
            )
    raise ValueError(f"Unknown outlier method: {method}")


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Overlap of two flagged sets: |A ∩ B| / |A ∪ B|."""
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def plot_summary(masks: dict[str, np.ndarray], y: pd.Series) -> None:
    """Share flagged and the readmission rate inside and outside each flagged set."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    names = [LABELS[m] for m in METHODS]
    shares = [masks[m].mean() * 100 for m in METHODS]
    bars = axes[0].bar(names, shares, color="steelblue")
    axes[0].bar_label(bars, fmt="%.1f %%", fontsize=9)
    axes[0].set_ylabel("Частка позначених спостережень, %")
    axes[0].set_title("Скільки спостережень позначає кожен метод")
    axes[0].tick_params(axis="x", labelsize=9)

    x = np.arange(len(METHODS))
    inside = [y[masks[m]].mean() * 100 for m in METHODS]
    outside = [y[~masks[m]].mean() * 100 for m in METHODS]
    axes[1].bar(x - 0.2, inside, width=0.4, color="crimson", label="серед викидів")
    axes[1].bar(x + 0.2, outside, width=0.4, color="grey", label="серед решти")
    axes[1].axhline(
        y.mean() * 100,
        color="black",
        linestyle="--",
        linewidth=1,
        label="у всій вибірці",
    )
    axes[1].set_xticks(x, names, fontsize=9)
    axes[1].set_ylabel("Частка повторних госпіталізацій, %")
    axes[1].set_title("Повторна госпіталізація <30 днів")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, LAB1 / "outliers_summary.png")


def plot_agreement(masks: dict[str, np.ndarray]) -> pd.DataFrame:
    """Pairwise Jaccard index between the flagged sets."""
    matrix = pd.DataFrame(
        [[jaccard(masks[a], masks[b]) for b in METHODS] for a in METHODS],
        index=[LABELS[m] for m in METHODS],
        columns=[LABELS[m] for m in METHODS],
    )
    fig, ax = plt.subplots(figsize=(6, 4.8))
    sns.heatmap(
        matrix, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, square=True, ax=ax
    )
    ax.set_title("Узгодженість методів (індекс Жаккара)")
    ax.tick_params(axis="x", rotation=20, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    save_figure(fig, LAB1 / "outliers_jaccard.png")
    return matrix


def plot_projection(X: pd.DataFrame, masks: dict[str, np.ndarray]) -> None:
    """The flagged rows of each method on a common 2-D PCA projection."""
    Z = StandardScaler().fit_transform(X)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(Z), size=min(8000, len(Z)), replace=False)
    pca = PCA(n_components=2, random_state=SEED).fit(Z)
    P = pca.transform(Z[idx])
    ratio = pca.explained_variance_ratio_ * 100

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharex=True, sharey=True)
    for ax, method in zip(axes.ravel(), METHODS):
        flagged = masks[method][idx]
        ax.scatter(
            P[~flagged, 0], P[~flagged, 1], s=3, color="lightgrey", label="звичайні"
        )
        ax.scatter(P[flagged, 0], P[flagged, 1], s=4, color="crimson", label="викиди")
        ax.set_title(f"{LABELS[method]}: {flagged.mean() * 100:.1f} %")
        ax.legend(fontsize=8, markerscale=3, loc="upper right")
    for ax in axes[1]:
        ax.set_xlabel(f"PC1 ({ratio[0]:.1f} % дисперсії)")
    for ax in axes[:, 0]:
        ax.set_ylabel(f"PC2 ({ratio[1]:.1f} % дисперсії)")
    fig.suptitle(
        "Викиди у проєкції на дві головні компоненти (8000 випадкових записів)"
    )
    fig.tight_layout()
    save_figure(fig, LAB1 / "outliers_pca.png")


def run_outliers(df: pd.DataFrame) -> None:
    """Compares the four detectors on the numeric features and writes outliers.json."""
    X = df[NUMERIC].astype(float)
    y = df[TARGET]

    masks = {m: outlier_mask(X, m) for m in METHODS}
    for m in METHODS:
        log.info("%-8s flags %5.2f%% of rows", m, masks[m].mean() * 100)

    plot_summary(masks, y)
    agreement = plot_agreement(masks)
    plot_projection(X, masks)

    per_feature = pd.DataFrame(
        {
            "iqr_pct": iqr_flags(X).mean() * 100,
            "zscore_pct": zscore_flags(X).mean() * 100,
            "zero_pct": (X == 0).mean() * 100,
        }
    )
    all_four = np.logical_and.reduce([masks[m] for m in METHODS])
    any_of = np.logical_or.reduce([masks[m] for m in METHODS])
    save_json(
        {
            "share_pct": {m: masks[m].mean() * 100 for m in METHODS},
            "count": {m: int(masks[m].sum()) for m in METHODS},
            "readmit_inside_pct": {m: y[masks[m]].mean() * 100 for m in METHODS},
            "readmit_outside_pct": {m: y[~masks[m]].mean() * 100 for m in METHODS},
            "jaccard": agreement.round(3).to_dict(),
            "per_feature": per_feature.round(2).to_dict(orient="index"),
            "flagged_by_all_pct": all_four.mean() * 100,
            "flagged_by_any_pct": any_of.mean() * 100,
        },
        LAB1 / "outliers.json",
    )
