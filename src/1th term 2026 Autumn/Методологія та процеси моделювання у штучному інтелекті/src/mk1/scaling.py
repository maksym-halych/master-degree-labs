"""Scaling and power transformations compared on the skewed counts (task 3).

Figure titles and axis labels stay in Ukrainian: they are report content, read
off the rendered figure, not code the reader of this module has to follow.
"""

import logging

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import kurtosis, normaltest, skew

from mk1.artifacts import save_figure, save_json
from mk1.config import LAB1
from mk1.data import NUMERIC
from mk1.preprocessing import make_scaler

log = logging.getLogger(__name__)

METHODS = ("raw", "standard", "robust", "yeo-johnson", "box-cox")
LABELS = {
    "raw": "вихідні",
    "standard": "StandardScaler",
    "robust": "RobustScaler",
    "yeo-johnson": "Yeo-Johnson",
    "box-cox": "Box-Cox (зсув до 1)",
}
SHOWN_FEATURES = ("num_medications", "number_inpatient", "number_emergency")
SHOWN_METHODS = ("raw", "robust", "yeo-johnson", "box-cox")


def transform_all(X: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The numeric features under every compared transformation, fitted on X."""
    result = {"raw": X}
    for method in METHODS[1:]:
        result[method] = pd.DataFrame(
            make_scaler(method).fit_transform(X), columns=X.columns, index=X.index
        )
    return result


def run_scaling(X_train: pd.DataFrame) -> None:
    """Measures skewness under each transformation and writes scaling.json."""
    X = X_train[NUMERIC].astype(float)
    transformed = transform_all(X)

    skewness = pd.DataFrame({m: transformed[m].apply(skew) for m in METHODS})
    excess = pd.DataFrame({m: transformed[m].apply(kurtosis) for m in METHODS})
    # Even a perfect power transform cannot make a count normal: a point mass
    # at zero stays a point mass. The test after Yeo-Johnson shows how far off.
    after = {c: normaltest(transformed["yeo-johnson"][c]).pvalue for c in NUMERIC}
    lambdas = make_scaler("yeo-johnson").fit(X).lambdas_

    fig, axes = plt.subplots(len(SHOWN_FEATURES), len(SHOWN_METHODS), figsize=(14, 8.5))
    for row, feature in enumerate(SHOWN_FEATURES):
        for col, method in enumerate(SHOWN_METHODS):
            ax = axes[row, col]
            ax.hist(transformed[method][feature], bins=40, color="steelblue")
            ax.set_title(
                f"{feature}: {LABELS[method]}\nасиметрія {skewness.loc[feature, method]:.2f}",
                fontsize=9,
            )
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.3)
    fig.suptitle("Вплив масштабування та степеневих перетворень на форму розподілу")
    fig.tight_layout()
    save_figure(fig, LAB1 / "scaling.png")

    save_json(
        {
            "skew": skewness.round(3).to_dict(orient="index"),
            "excess_kurtosis": excess.round(2).to_dict(orient="index"),
            "yeo_johnson_lambda": dict(zip(NUMERIC, lambdas.round(3))),
            "dagostino_p_after_yeo_johnson": after,
            "robust_scale": dict(zip(NUMERIC, make_scaler("robust").fit(X).scale_)),
        },
        LAB1 / "scaling.json",
    )
    log.info("Mean |skew|: %s", skewness.abs().mean().round(2).to_dict())
