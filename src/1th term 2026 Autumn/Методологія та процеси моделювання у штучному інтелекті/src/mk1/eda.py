"""Advanced exploratory data analysis (task 2).

Distributions and normality tests, Pearson and Spearman correlations,
multicollinearity (VIF, Cramér's V), missing values and cardinality. Writes the
figures and eda.json into the lab's artifact directory.

Figure titles and axis labels stay in Ukrainian: they are report content, read
off the rendered figure, not code the reader of this module has to follow.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, kurtosis, normaltest, probplot, shapiro, skew
from sklearn.linear_model import LinearRegression

from mk1.artifacts import save_figure, save_json
from mk1.config import LAB1, SEED, SHAPIRO_SAMPLE
from mk1.data import (
    AGE_ORDER,
    CATEGORICAL,
    DISGUISED_MISSING,
    HIGH_CARD,
    NUMERIC,
    TARGET,
)
from mk1.features import add_diag_groups

log = logging.getLogger(__name__)

# Columns missing in fewer rows than this are left out of the readmission
# comparison: their "missing" group is too small for a stable rate.
MIN_MISSING_PCT_SHOWN = 0.3

# The categorical features shown in the association matrix: the ones with a
# readable number of levels plus the ICD-9 chapter of the primary diagnosis.
ASSOCIATION_COLUMNS = [
    "race",
    "gender",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "payer_code",
    "medical_specialty",
    "max_glu_serum",
    "A1Cresult",
    "metformin",
    "insulin",
    "change",
    "diabetesMed",
    "diag_1_group",
    TARGET,
]


def class_balance(raw: pd.DataFrame, y: pd.Series) -> dict:
    """The three source classes of `readmitted` and the binary target built from them."""
    source = raw["readmitted"].value_counts().reindex(["NO", ">30", "<30"])
    binary = y.value_counts().sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    bars = axes[0].bar(
        ["немає", "> 30 днів", "< 30 днів"],
        source.to_numpy(),
        color=["grey", "grey", "crimson"],
    )
    axes[0].bar_label(
        bars, labels=[f"{v / source.sum() * 100:.1f} %" for v in source], fontsize=9
    )
    axes[0].set_title("Вихідна змінна readmitted (101 766 госпіталізацій)")
    axes[0].set_ylabel("Кількість")

    bars = axes[1].bar(
        ["0: інше", "1: < 30 днів"], binary.to_numpy(), color=["grey", "crimson"]
    )
    axes[1].bar_label(
        bars, labels=[f"{v / binary.sum() * 100:.2f} %" for v in binary], fontsize=9
    )
    axes[1].set_title(
        f"Бінарна ціль після очищення ({len(y):,} пацієнтів)".replace(",", " ")
    )
    axes[1].set_ylabel("Кількість")
    fig.tight_layout()
    save_figure(fig, LAB1 / "class_balance.png")

    return {
        "source_counts": source.to_dict(),
        "binary_counts": {str(k): int(v) for k, v in binary.items()},
        "positive_pct": y.mean() * 100,
        "imbalance_ratio": binary[0] / binary[1],
    }


def missingness(raw: pd.DataFrame, df: pd.DataFrame) -> dict:
    """Missing share per column and whether being missing relates to the target."""
    share = df.isna().mean().mul(100)
    share = share[share > 0].sort_values(ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = [
        "darkorange" if c in DISGUISED_MISSING else "steelblue" for c in share.index
    ]
    bars = axes[0].barh(share.index[::-1], share.to_numpy()[::-1], color=colors[::-1])
    axes[0].bar_label(bars, fmt="%.2f %%", fontsize=8)
    axes[0].set_xlabel("Частка пропусків, %")
    axes[0].set_title("Пропуски після очищення")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="steelblue"),
        plt.Rectangle((0, 0), 1, 1, color="darkorange"),
    ]
    axes[0].legend(
        handles,
        ["явні («?»)", "приховані (коди NULL / Not Mapped)"],
        fontsize=8,
        loc="lower right",
    )

    y = df[TARGET]
    rows = []
    for column in share.index:
        missing = df[column].isna()
        table = pd.crosstab(missing, y)
        chi2, p, _, _ = chi2_contingency(table)
        rows.append(
            {
                "column": column,
                "missing_pct": share[column],
                "readmit_missing_pct": y[missing].mean() * 100,
                "readmit_observed_pct": y[~missing].mean() * 100,
                "chi2": chi2,
                "p_value": p,
            }
        )
    informative = pd.DataFrame(rows).set_index("column")

    # A rate over a handful of rows is noise: diag_1 is missing in 10 rows.
    shown = informative[informative["missing_pct"] >= MIN_MISSING_PCT_SHOWN]
    x = np.arange(len(shown))
    axes[1].bar(
        x - 0.2,
        shown["readmit_missing_pct"],
        width=0.4,
        color="crimson",
        label="значення пропущене",
    )
    axes[1].bar(
        x + 0.2,
        shown["readmit_observed_pct"],
        width=0.4,
        color="grey",
        label="значення наявне",
    )
    axes[1].axhline(y.mean() * 100, color="black", linestyle="--", linewidth=1)
    axes[1].set_xticks(x, shown.index, rotation=35, ha="right", fontsize=8)
    axes[1].set_ylabel("Частка повторних госпіталізацій, %")
    axes[1].set_title("Чи несе сам факт пропуску інформацію?")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, LAB1 / "missingness.png")

    raw_share = raw.isna().mean().mul(100)
    return {
        "raw_missing_pct": raw_share[raw_share > 0].round(2).to_dict(),
        "clean_missing": informative.round(4).to_dict(orient="index"),
    }


def distributions(X: pd.DataFrame) -> None:
    """Histograms of the numeric features; zero-inflated ones on a log axis."""
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5))
    for ax, column in zip(axes.ravel(), NUMERIC):
        values = X[column]
        bins = (
            np.arange(values.min(), values.max() + 2) - 0.5
            if values.nunique() < 60
            else 50
        )
        ax.hist(values, bins=bins, color="steelblue")
        if (values == 0).mean() > 0.5:
            ax.set_yscale("log")
        ax.set_title(f"{column}\nасиметрія {skew(values):.2f}", fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle(
        "Розподіли числових ознак (логарифмічна вісь Y для ознак із переважанням нулів)"
    )
    fig.tight_layout()
    save_figure(fig, LAB1 / "numeric_distributions.png")


def qq_plots(X: pd.DataFrame) -> None:
    """Quantile-quantile plots of the standardized features against N(0, 1)."""
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5))
    for ax, column in zip(axes.ravel(), NUMERIC):
        z = (X[column] - X[column].mean()) / X[column].std()
        (theoretical, ordered), _ = probplot(z, dist="norm")
        ax.scatter(theoretical, ordered, s=2, color="steelblue")
        lim = [theoretical.min(), theoretical.max()]
        ax.plot(lim, lim, color="crimson", linewidth=1)
        ax.set_title(column, fontsize=9)
        ax.set_xlabel("Теоретичні квантилі N(0, 1)", fontsize=8)
        ax.set_ylabel("Вибіркові квантилі", fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Q-Q графіки стандартизованих числових ознак")
    fig.tight_layout()
    save_figure(fig, LAB1 / "qq_plots.png")


def normality(X: pd.DataFrame) -> dict:
    """
    Shapiro-Wilk and D'Agostino-Pearson tests of every numeric feature.

    Shapiro-Wilk runs on a random subsample: above 5000 observations scipy no
    longer guarantees its p-value. D'Agostino's K² has no such limit and runs
    on the full column.
    """
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X), size=min(SHAPIRO_SAMPLE, len(X)), replace=False)
    result = {}
    for column in NUMERIC:
        values = X[column].to_numpy(dtype=float)
        w, p_w = shapiro(values[idx])
        k2, p_k2 = normaltest(values)
        result[column] = {
            "skew": skew(values),
            "excess_kurtosis": kurtosis(values),
            "shapiro_w": w,
            "shapiro_p": p_w,
            "dagostino_k2": k2,
            "dagostino_p": p_k2,
        }
    return result


def correlations(df: pd.DataFrame) -> dict:
    """Pearson and Spearman matrices of the numeric features, age and the target."""
    data = df[NUMERIC].astype(float)
    data["age"] = df["age"].map({a: i for i, a in enumerate(AGE_ORDER)})
    data[TARGET] = df[TARGET]
    pearson = data.corr(method="pearson")
    spearman = data.corr(method="spearman")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.8))
    for ax, matrix, title in (
        (axes[0], pearson, "Пірсона"),
        (axes[1], spearman, "Спірмена"),
    ):
        mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
        sns.heatmap(
            matrix,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            cbar_kws={"shrink": 0.7},
            annot_kws={"size": 7},
            ax=ax,
        )
        ax.set_title(f"Кореляція {title}")
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    save_figure(fig, LAB1 / "correlation.png")

    gap = (spearman - pearson).abs()
    pair = gap.where(~np.eye(len(gap), dtype=bool), 0.0).stack().idxmax()
    return {
        "pearson": pearson.round(3).to_dict(),
        "spearman": spearman.round(3).to_dict(),
        "largest_rank_gap": {"pair": list(pair), "gap": gap.loc[pair]},
    }


def vif(X: pd.DataFrame) -> dict[str, float]:
    """
    Variance inflation factor of every column: 1 / (1 - R²) of its regression on the rest.

    Args:
        X: Numeric features without missing values.

    Returns:
        VIF per column; infinity for a column the others reproduce exactly.
    """
    values = X.to_numpy(dtype=float)
    result = {}
    for j, column in enumerate(X.columns):
        others = np.delete(values, j, axis=1)
        r2 = LinearRegression().fit(others, values[:, j]).score(others, values[:, j])
        result[column] = float("inf") if r2 >= 1.0 else 1.0 / (1.0 - r2)
    return result


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    """Bias-corrected Cramér's V (Bergsma, 2013) of two categorical variables."""
    table = pd.crosstab(a, b)
    n = table.to_numpy().sum()
    chi2 = chi2_contingency(table, correction=False)[0]
    r, k = table.shape
    phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    k_corr = k - (k - 1) ** 2 / (n - 1)
    denominator = min(k_corr - 1, r_corr - 1)
    return float(np.sqrt(phi2 / denominator)) if denominator > 0 else 0.0


def associations(df: pd.DataFrame) -> dict:
    """Cramér's V between the categorical features — multicollinearity among categories."""
    data = add_diag_groups(df)[ASSOCIATION_COLUMNS].astype("string").fillna("Missing")
    columns = ASSOCIATION_COLUMNS
    matrix = pd.DataFrame(np.eye(len(columns)), index=columns, columns=columns)
    for i, a in enumerate(columns):
        for b in columns[i + 1 :]:
            matrix.loc[a, b] = matrix.loc[b, a] = cramers_v(data[a], data[b])

    fig, ax = plt.subplots(figsize=(10, 8.5))
    mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
    sns.heatmap(
        matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="Purples",
        vmin=0,
        vmax=1,
        square=True,
        cbar_kws={"shrink": 0.7},
        annot_kws={"size": 7},
        ax=ax,
    )
    ax.set_title("Зв'язок між категоріальними ознаками (V Крамера)")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    save_figure(fig, LAB1 / "cramers_v.png")

    pairs = (
        matrix.where(~mask & ~np.eye(len(columns), dtype=bool))
        .stack()
        .sort_values(ascending=False)
    )
    return {
        "top_pairs": [{"a": a, "b": b, "v": v} for (a, b), v in pairs.head(8).items()],
        "with_target": matrix[TARGET]
        .drop(TARGET)
        .sort_values(ascending=False)
        .round(4)
        .to_dict(),
    }


def cardinality(df: pd.DataFrame) -> dict:
    """Number of levels per categorical feature and how concentrated the codes are."""
    levels = df[CATEGORICAL].nunique().sort_values()
    codes = df["diag_1"].value_counts(normalize=True)
    coverage = codes.cumsum().to_numpy() * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["crimson" if c in HIGH_CARD else "steelblue" for c in levels.index]
    bars = axes[0].barh(levels.index, levels.to_numpy(), color=colors)
    axes[0].bar_label(bars, fontsize=7)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Кількість унікальних значень (логарифмічна шкала)")
    axes[0].set_title("Кардинальність категоріальних ознак")
    axes[0].tick_params(axis="y", labelsize=7)

    axes[1].plot(np.arange(1, len(coverage) + 1), coverage, color="crimson")
    for top in (10, 50):
        axes[1].axvline(top, color="grey", linestyle=":", linewidth=1)
        axes[1].annotate(
            f"топ-{top}: {coverage[top - 1]:.0f} %",
            (top, coverage[top - 1]),
            textcoords="offset points",
            xytext=(6, -14),
            fontsize=8,
        )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Кількість найчастіших кодів diag_1 (логарифмічна шкала)")
    axes[1].set_ylabel("Охоплення записів, %")
    axes[1].set_title(f"Довгий хвіст кодів основного діагнозу ({len(codes)} кодів)")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    save_figure(fig, LAB1 / "cardinality.png")

    singletons = int((df["diag_1"].value_counts() < 10).sum())
    return {
        "levels": levels.to_dict(),
        "diag_1_top10_pct": coverage[9],
        "diag_1_top50_pct": coverage[49],
        "diag_1_codes_under_10_rows": singletons,
    }


def run_eda(raw: pd.DataFrame, df: pd.DataFrame, steps: dict[str, int]) -> None:
    """Runs every part of the exploratory analysis and writes eda.json."""
    log.info("Cleaned dataset: %d rows, %d columns", *df.shape)
    X = df[NUMERIC].astype(float)

    balance = class_balance(raw, df[TARGET])
    missing = missingness(raw, df)
    distributions(X)
    qq_plots(X)
    tests = normality(X)
    corr = correlations(df)
    associated = associations(df)
    levels = cardinality(df)

    save_json(
        {
            "cleaning_steps": steps,
            "n_rows": len(df),
            "n_features": df.shape[1] - 1,
            "class_balance": balance,
            "missingness": missing,
            "describe": X.describe().round(3).to_dict(),
            "zero_pct": (X == 0).mean().mul(100).round(2).to_dict(),
            "normality": tests,
            "correlation": corr,
            "vif": vif(X),
            "associations": associated,
            "cardinality": levels,
        },
        LAB1 / "eda.json",
    )
    log.info("EDA finished, artifacts in %s", LAB1)
