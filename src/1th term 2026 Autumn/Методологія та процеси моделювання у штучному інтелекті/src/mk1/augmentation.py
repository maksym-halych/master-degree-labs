"""Synthesis of minority-class observations (task 4).

Every sampler here runs inside an imbalanced-learn pipeline, which calls it
during fit only: a cross-validation fold or the test set is never resampled,
so the reported metrics are measured on real, untouched patients.

Figure titles and axis labels stay in Ukrainian: they are report content, read
off the rendered figure, not code the reader of this module has to follow.
"""

import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn import FunctionSampler
from imblearn.base import BaseSampler
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import ADASYN, SMOTE, SMOTENC, RandomOverSampler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline

from mk1.artifacts import save_figure, save_json
from mk1.config import LAB1, SEED
from mk1.data import CATEGORICAL, NUMERIC
from mk1.features import DIAG_GROUPS, FeatureEngineer
from mk1.outliers import outlier_mask
from mk1.preprocessing import CategoricalImputer, make_preprocessor

log = logging.getLogger(__name__)

NOISE_SIGMA = 0.1

# Neighbours consulted when judging where a synthetic point landed.
TERRITORY_K = 5


def gaussian_noise_oversample(
    X: np.ndarray, y: np.ndarray, sigma: float = NOISE_SIGMA, seed: int = SEED
) -> tuple[np.ndarray, np.ndarray]:
    """
    Balance the classes with noisy copies of minority rows.

    Each copy perturbs the continuous columns by N(0, (sigma · column std)²);
    binary columns — the one-hot indicators — are copied unchanged, since a
    fractional category has no meaning. Applied after the train/test split, to
    the training rows only.

    Args:
        X: Preprocessed training matrix.
        y: Binary labels.
        sigma: Noise scale relative to each column's standard deviation.
        seed: Random seed.

    Returns:
        The original rows followed by the synthetic ones, and their labels.
    """
    rng = np.random.default_rng(seed)
    minority = np.flatnonzero(y == 1)
    n_new = int((y == 0).sum() - minority.size)
    source = X[rng.choice(minority, size=n_new, replace=True)]

    continuous = np.array([np.unique(X[:, j]).size > 2 for j in range(X.shape[1])])
    noise = rng.normal(0.0, 1.0, size=(n_new, int(continuous.sum())))
    synthetic = source.copy()
    synthetic[:, continuous] += sigma * X[:, continuous].std(axis=0) * noise
    return np.vstack([X, synthetic]), np.concatenate([y, np.ones(n_new, dtype=y.dtype)])


def drop_outliers(
    X: pd.DataFrame, y: pd.Series, method: str
) -> tuple[pd.DataFrame, pd.Series]:
    """Remove the training rows a detector flags on the base numeric features."""
    keep = ~outlier_mask(X[NUMERIC].astype(float), method)
    return X[keep], y[keep]


def smotenc_columns(domain: bool) -> list[str]:
    """The columns SMOTENC must treat as categories rather than as numbers."""
    return [*CATEGORICAL, "age", *(DIAG_GROUPS if domain else [])]


def make_sampler(
    name: str, domain: bool = True, sigma: float = NOISE_SIGMA
) -> BaseSampler | None:
    """
    Build one of the compared resampling strategies.

    Args:
        name: "none", "class-weight", "ros", "smote", "smotenc", "adasyn",
            "noise" or "smote-enn". The first two need no sampler: class
            weights act inside the model.
        domain: Whether the domain features are present (SMOTENC needs to
            know every categorical column).
        sigma: Noise scale of the "noise" sampler.

    Returns:
        An unfitted sampler, or None.

    Raises:
        ValueError: If the name is unknown.
    """
    match name:
        case "none" | "class-weight":
            return None
        case "ros":
            return RandomOverSampler(random_state=SEED)
        case "smote":
            return SMOTE(k_neighbors=5, random_state=SEED)
        case "smotenc":
            return SMOTENC(
                categorical_features=smotenc_columns(domain),
                k_neighbors=5,
                random_state=SEED,
            )
        case "adasyn":
            return ADASYN(n_neighbors=5, random_state=SEED)
        case "noise":
            return FunctionSampler(
                func=gaussian_noise_oversample, validate=False, kw_args={"sigma": sigma}
            )
        case "smote-enn":
            return SMOTEENN(
                smote=SMOTE(k_neighbors=5, random_state=SEED), random_state=SEED
            )
    raise ValueError(f"Unknown sampler: {name}")


def resample(
    name: str,
    frame: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    preprocessor: ColumnTransformer,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply one sampler; over-samplers return the real rows first, then the synthetic ones.

    SMOTENC works on the categories themselves, before encoding, so it gets
    the frame and its output is encoded afterwards by the same preprocessor.
    Every other sampler works on the already encoded matrix X.
    """
    if name != "smotenc":
        return make_sampler(name).fit_resample(X, y)
    res_frame, y_res = smotenc_frame(frame, y)
    return preprocessor.transform(res_frame), np.asarray(y_res)


def smotenc_frame(
    frame: pd.DataFrame, y: np.ndarray
) -> tuple[pd.DataFrame, np.ndarray]:
    """SMOTENC on the raw categories and numbers; real rows first, then synthetic."""
    numeric = [c for c in frame.columns if frame[c].dtype.kind in "if"]
    res_frame, y_res = make_sampler("smotenc").fit_resample(
        frame[[*smotenc_columns(True), *numeric]], y
    )
    return res_frame, np.asarray(y_res)


# The categorical columns most associated with the target, plus the ones
# where SMOTENC's majority vote over neighbours shows most clearly.
PROFILE_CATEGORIES = (
    "discharge_disposition_id",
    "medical_specialty",
    "payer_code",
    "diag_1_group",
    "A1Cresult",
    "race",
)


def smotenc_profile(frame: pd.DataFrame, y: np.ndarray) -> dict:
    """
    Compare SMOTENC's synthetic minority rows with the real minority rows.

    SMOTENC gives a synthetic row the most frequent category among its
    neighbours, and imbalanced-learn casts interpolated counts back to the
    integer dtype of their column. Both pull the synthetic rows towards the
    typical patient, which in this dataset is a majority-class patient.
    """
    res_frame, _ = smotenc_frame(frame, y)
    real = frame[y == 1]
    synthetic = res_frame.iloc[len(frame) :]
    categories = {}
    for column in PROFILE_CATEGORIES:
        top = real[column].value_counts(normalize=True)
        categories[column] = {
            "top": top.index[0],
            "real_pct": top.iloc[0] * 100,
            "synthetic_pct": (synthetic[column] == top.index[0]).mean() * 100,
        }
    return {
        "categories": categories,
        "number_inpatient_mean": {
            "real_minority": real["number_inpatient"].mean(),
            "synthetic": synthetic["number_inpatient"].mean(),
            "real_majority": frame[y == 0]["number_inpatient"].mean(),
        },
        "number_inpatient_dtype": str(synthetic["number_inpatient"].dtype),
        # Derived features are interpolated independently of the counts they
        # were computed from, so a synthetic row stops agreeing with itself —
        # something no real row ever does.
        "inconsistent_pct": {
            "meds_per_day": inconsistent_share(
                synthetic["meds_per_day"],
                synthetic["num_medications"] / synthetic["time_in_hospital"],
            ),
            "total_visits": inconsistent_share(
                synthetic["total_visits"],
                synthetic["number_outpatient"]
                + synthetic["number_emergency"]
                + synthetic["number_inpatient"],
            ),
        },
    }


def inconsistent_share(derived: pd.Series, recomputed: pd.Series) -> float:
    """Percentage of rows where a derived feature no longer matches its formula."""
    return float(((derived - recomputed).abs() > 1e-6).mean() * 100)


def majority_territory(
    frame: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    preprocessor: ColumnTransformer,
    rng: np.random.Generator,
) -> dict[str, float]:
    """
    Share of majority-class patients among the nearest neighbours of minority points.

    A synthetic minority point surrounded by majority patients sits in the
    overlap zone: it teaches the model that a typical non-readmitted profile
    may be a readmission. The minority is split in halves: the samplers see
    one half, the neighbours are searched among the majority plus the other
    half. Otherwise a synthetic point's nearest neighbour would mostly be the
    very patient it was generated from, which would flatter every sampler.

    Returns:
        The share for real minority points of the first half (the reference),
        for each sampler's synthetic points, and the majority's share of the
        searched set (what a point placed at random would score).
    """
    minority = rng.permutation(np.flatnonzero(y == 1))
    seen, held_out = np.array_split(minority, 2)
    majority = np.flatnonzero(y == 0)
    generate = np.concatenate([majority, seen])
    search = np.concatenate([majority, held_out])

    index = NearestNeighbors(n_neighbors=TERRITORY_K).fit(X[search])
    labels = y[search]

    def share(points: np.ndarray) -> float:
        pick = rng.choice(len(points), size=min(2000, len(points)), replace=False)
        _, neighbours = index.kneighbors(points[pick])
        return float((labels[neighbours] == 0).mean())

    result = {"base_rate": float((labels == 0).mean()), "real_minority": share(X[seen])}
    for name in ("ros", "smote", "smotenc", "adasyn", "noise"):
        X_res, _ = resample(
            name, frame.iloc[generate], X[generate], y[generate], preprocessor
        )
        result[name] = share(X_res[len(generate) :])
    return result


def fractional_share(X_real: np.ndarray, synthetic: np.ndarray) -> float:
    """Share of synthetic rows holding a fraction in a column that is 0/1 in real data."""
    binary = np.array(
        [np.isin(X_real[:, j], (0.0, 1.0)).all() for j in range(X_real.shape[1])]
    )
    return float((~np.isin(synthetic[:, binary], (0.0, 1.0))).any(axis=1).mean())


def detectability(
    real: np.ndarray, synthetic: np.ndarray, rng: np.random.Generator
) -> float:
    """
    How well a classifier tells synthetic minority rows from real ones (ROC-AUC).

    Adversarial validation: 0.5 means the synthetic rows are indistinguishable
    from real patients. Anything a classifier can use to spot them, a model
    trained on the resampled data can use too — and on real test patients
    that signal is never there.
    """
    n = min(len(real), len(synthetic), 5000)
    X = np.vstack(
        [
            real[rng.choice(len(real), size=n, replace=False)],
            synthetic[rng.choice(len(synthetic), size=n, replace=False)],
        ]
    )
    y = np.repeat([0, 1], n)
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    detector = HistGradientBoostingClassifier(max_iter=100, random_state=SEED)
    return float(cross_val_score(detector, X, y, cv=folds, scoring="roc_auc").mean())


def run_augmentation(X_train: pd.DataFrame, y_train: pd.Series) -> None:
    """Resamples the preprocessed training set with every method and writes augmentation.json."""
    rng = np.random.default_rng(SEED)
    prepare = make_pipeline(FeatureEngineer(), CategoricalImputer())
    frame = prepare.fit_transform(X_train)
    preprocessor = make_preprocessor().fit(frame, y_train)
    X = preprocessor.transform(frame)
    y = y_train.to_numpy()
    log.info("Preprocessed training matrix: %d x %d", *X.shape)

    counts, seconds, synthetic = {}, {}, {}
    for name in ("ros", "smote", "smotenc", "adasyn", "noise", "smote-enn"):
        started = time.perf_counter()
        X_res, y_res = resample(name, frame, X, y, preprocessor)
        seconds[name] = time.perf_counter() - started
        counts[name] = {"0": int((y_res == 0).sum()), "1": int((y_res == 1).sum())}
        if name != "smote-enn":
            synthetic[name] = X_res[len(y) :]
        log.info("%-9s %6.1fs  class counts %s", name, seconds[name], counts[name])

    fractions = {n: fractional_share(X, points) for n, points in synthetic.items()}
    # ROS rows are exact copies of real ones, so a detector has nothing to
    # find: the same row appears under both labels and the score is noise.
    detection = {
        n: detectability(X[y == 1], points, rng)
        for n, points in synthetic.items()
        if n != "ros"
    }
    log.info(
        "Rows with fractional categories: %s",
        {k: round(v, 3) for k, v in fractions.items()},
    )
    log.info(
        "Detectability (ROC-AUC): %s", {k: round(v, 3) for k, v in detection.items()}
    )

    profile = smotenc_profile(frame, y)
    log.info("SMOTENC profile: %s", profile)

    territory = majority_territory(frame, X, y, preprocessor, rng)
    log.info(
        "Majority share among neighbours: %s",
        {k: round(v, 3) for k, v in territory.items()},
    )

    plot_counts(y, counts)
    plot_projection(X, y, synthetic, rng)
    plot_territory(territory, detection)
    save_json(
        {
            "n_features": X.shape[1],
            "original_counts": {"0": int((y == 0).sum()), "1": int((y == 1).sum())},
            "counts": counts,
            "seconds": seconds,
            "majority_territory": territory,
            "fractional_share": fractions,
            "detectability_auc": detection,
            "smotenc_profile": profile,
            "territory_k": TERRITORY_K,
            "noise_sigma": NOISE_SIGMA,
        },
        LAB1 / "augmentation.json",
    )


LABELS = {
    "ros": "ROS",
    "smote": "SMOTE",
    "smotenc": "SMOTENC",
    "adasyn": "ADASYN",
    "noise": "Гаусів шум",
    "smote-enn": "SMOTE-ENN",
}


def plot_counts(y: np.ndarray, counts: dict) -> None:
    """Class sizes before and after each sampler."""
    names = ["original", *counts]
    zeros = [int((y == 0).sum()), *[counts[n]["0"] for n in counts]]
    ones = [int((y == 1).sum()), *[counts[n]["1"] for n in counts]]
    labels = ["без змін", *[LABELS[n] for n in counts]]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(x - 0.2, zeros, width=0.4, color="grey", label="клас 0")
    bars = ax.bar(x + 0.2, ones, width=0.4, color="crimson", label="клас 1 (< 30 днів)")
    ax.bar_label(bars, fontsize=7)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Кількість спостережень")
    ax.set_title("Розмір класів тренувальної вибірки після передискретизації")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, LAB1 / "augmentation_counts.png")


def plot_projection(
    X: np.ndarray, y: np.ndarray, synthetic: dict, rng: np.random.Generator
) -> None:
    """Real classes and synthetic minority points on the real data's PCA plane."""
    pca = PCA(n_components=2, random_state=SEED).fit(X)
    majority = pca.transform(
        X[y == 0][rng.choice(int((y == 0).sum()), size=4000, replace=False)]
    )
    minority = pca.transform(X[y == 1])
    shown = ("smote", "smotenc", "adasyn", "noise")
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True)
    for ax, name in zip(axes.ravel(), shown):
        points = synthetic[name]
        new = pca.transform(
            points[rng.choice(len(points), size=min(3000, len(points)), replace=False)]
        )
        ax.scatter(
            majority[:, 0],
            majority[:, 1],
            s=2,
            color="lightgrey",
            label="клас 0 (реальні)",
        )
        ax.scatter(
            new[:, 0],
            new[:, 1],
            s=2,
            color="darkorange",
            alpha=0.6,
            label="клас 1 (синтетичні)",
        )
        ax.scatter(
            minority[:, 0],
            minority[:, 1],
            s=2,
            color="crimson",
            alpha=0.6,
            label="клас 1 (реальні)",
        )
        ax.set_title(LABELS[name])
        ax.legend(fontsize=7, markerscale=4, loc="upper right")
    fig.suptitle("Синтетичні спостереження у проєкції на дві головні компоненти")
    fig.tight_layout()
    save_figure(fig, LAB1 / "augmentation_pca.png")


def plot_territory(territory: dict, detection: dict) -> None:
    """Where the synthetic points land, and how easily they can be told from real ones."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    names = [n for n in territory if n != "base_rate"]
    labels = ["реальні\nміноритарні", *[LABELS[n] for n in names[1:]]]
    values = [territory[n] * 100 for n in names]
    colors = ["crimson", *["darkorange"] * (len(names) - 1)]
    bars = axes[0].bar(labels, values, color=colors)
    axes[0].bar_label(bars, fmt="%.1f", fontsize=8)
    axes[0].axhline(
        territory["base_rate"] * 100,
        color="black",
        linestyle="--",
        linewidth=1,
        label="випадкова точка",
    )
    axes[0].set_ylim(min(values) - 10, 100)
    axes[0].set_ylabel(f"Частка класу 0 серед {TERRITORY_K} сусідів, %")
    axes[0].set_title("Оточення реальних і синтетичних точок класу 1")
    axes[0].legend(fontsize=8, loc="lower right")
    axes[0].tick_params(axis="x", labelsize=8)

    names = list(detection)
    bars = axes[1].bar(
        [LABELS[n] for n in names], [detection[n] for n in names], color="darkorange"
    )
    axes[1].bar_label(bars, fmt="%.3f", fontsize=8)
    axes[1].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[1].text(-0.4, 0.51, "0,5: не відрізнити від реальних", fontsize=8)
    axes[1].set_ylim(0.4, 1.08)
    axes[1].set_ylabel("ROC-AUC класифікатора «реальна / синтетична»")
    axes[1].set_title("Чи можна відрізнити синтетичні точки від реальних")
    axes[1].tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    save_figure(fig, LAB1 / "augmentation_territory.png")
