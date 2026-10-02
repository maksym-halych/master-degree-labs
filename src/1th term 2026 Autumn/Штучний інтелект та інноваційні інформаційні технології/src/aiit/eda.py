"""Exploratory data analysis (tasks 3-4).

Produces the figures and the eda.json summary in the artifact directory
(config.EDA). The EDA figures are used by the lab 1 report, which is why the
stage belongs to it.

Figure titles and axis labels stay in Ukrainian: they are report content, read
off the rendered figure, not code the reader of this module has to follow.
"""

import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from aiit.config import EDA, SEED
from aiit.data import load_raw, target_bins
from aiit.figures import save_figure

log = logging.getLogger(__name__)

OUT = EDA


def feature_distributions(X, y) -> None:
    """Histograms of every feature and of the target."""
    df = X.copy()
    df["MedHouseVal"] = y

    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    for ax, col in zip(axes.ravel(), df.columns):
        ax.hist(df[col], bins=60, color="steelblue")
        ax.set_title(col, fontsize=10)
        ax.grid(alpha=0.3)

    fig.suptitle("Розподіл ознак та цільової змінної")
    fig.tight_layout()
    save_figure(fig, OUT / "distributions.png")
    plt.close(fig)


def correlation(X, y) -> dict:
    """Pearson correlation matrix."""
    df = X.copy()
    df["MedHouseVal"] = y
    corr = df.corr()

    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Кореляційна матриця")
    fig.tight_layout()
    save_figure(fig, OUT / "correlation.png")
    plt.close(fig)

    return corr["MedHouseVal"].drop("MedHouseVal").to_dict()


def target_imbalance(y) -> dict:
    """Analysis of the skew in the target distribution.

    The task is a regression, so there is no class imbalance. What we analyse
    instead are two effects: the skew across the quantile groups, and the
    artificial cap on the target at 5.00001.
    """
    bins = target_bins(y)
    counts = bins.value_counts().sort_index()
    capped = int((y >= 5.0).sum())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(counts.index.astype(str), counts.to_numpy(), color="steelblue")
    axes[0].set_xlabel("Квантильна група цільової змінної")
    axes[0].set_ylabel("Кількість спостережень")
    axes[0].set_title("Наповненість квантильних груп")

    axes[1].hist(y, bins=80, color="darkorange")
    axes[1].axvline(
        5.0,
        color="crimson",
        linestyle="--",
        label=f"обмеження зверху: {capped} записів",
    )
    axes[1].set_xlabel("MedHouseVal (×100 тис. \\$)")
    axes[1].set_ylabel("Кількість спостережень")
    axes[1].set_title("Перекіс та ефект «стелі»")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    save_figure(fig, OUT / "target_imbalance.png")
    plt.close(fig)

    return {
        "bin_counts": {str(k): int(v) for k, v in counts.items()},
        "capped_at_5": capped,
        "capped_share_pct": round(capped / len(y) * 100, 2),
        "skew": float(y.skew()),
    }


def pca_and_tsne(X, y) -> dict:
    """PCA (dimensionality reduction) and t-SNE (structure visualisation)."""
    Xs = StandardScaler().fit_transform(X)

    pca_full = PCA(random_state=SEED).fit(Xs)
    cum = np.cumsum(pca_full.explained_variance_ratio_)
    n_90 = int(np.searchsorted(cum, 0.90) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(np.arange(1, len(cum) + 1), cum, marker="o")
    axes[0].axhline(0.90, color="crimson", linestyle="--", label="90 % дисперсії")
    axes[0].set_xlabel("Кількість головних компонент")
    axes[0].set_ylabel("Накопичена пояснена дисперсія")
    axes[0].set_title("PCA: пояснена дисперсія")
    axes[0].legend(fontsize=8)

    # t-SNE over the full sample is far too slow — take a random subsample.
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(Xs), size=3000, replace=False)
    emb = TSNE(
        n_components=2, perplexity=30, init="pca", random_state=SEED
    ).fit_transform(Xs[idx])

    sc = axes[1].scatter(
        emb[:, 0], emb[:, 1], c=y.to_numpy()[idx], s=5, cmap="viridis", alpha=0.7
    )
    axes[1].set_title("t-SNE (3000 випадкових записів)")
    axes[1].set_xlabel("t-SNE 1")
    axes[1].set_ylabel("t-SNE 2")
    fig.colorbar(sc, ax=axes[1], label="MedHouseVal")

    fig.tight_layout()
    save_figure(fig, OUT / "pca_tsne.png")
    plt.close(fig)

    return {
        "explained_variance_ratio": [
            round(v, 4) for v in pca_full.explained_variance_ratio_
        ],
        "components_for_90pct": n_90,
    }


def run_eda() -> None:
    """Runs every stage of the exploratory analysis and writes eda.json."""
    OUT.mkdir(parents=True, exist_ok=True)

    X, y = load_raw()
    log.info("Dataset size: %d records, %d features", X.shape[0], X.shape[1])

    feature_distributions(X, y)
    corr = correlation(X, y)
    imbalance = target_imbalance(y)
    pca_info = pca_and_tsne(X, y)

    summary = {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "features": list(X.columns),
        "describe": X.describe().round(3).to_dict(),
        "target_correlation": {k: round(v, 4) for k, v in corr.items()},
        "target_imbalance": imbalance,
        "pca": pca_info,
    }
    (OUT / "eda.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info("EDA finished, artifacts in: %s", OUT)
