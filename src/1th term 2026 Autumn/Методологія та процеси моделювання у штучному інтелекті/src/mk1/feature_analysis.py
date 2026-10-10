"""How informative and how redundant the engineered features are (task 4).

Figure titles and axis labels stay in Ukrainian: they are report content, read
off the rendered figure, not code the reader of this module has to follow.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import PolynomialFeatures, PowerTransformer

from mk1.artifacts import save_figure, save_json
from mk1.config import LAB1, SEED
from mk1.data import NUMERIC
from mk1.eda import cramers_v, vif
from mk1.features import AGGREGATE_NUMERIC, DOMAIN_NUMERIC, FeatureEngineer

log = logging.getLogger(__name__)

ENGINEERED = [*DOMAIN_NUMERIC, *AGGREGATE_NUMERIC]
VIF_DISPLAY_CAP = 1000.0


def run_feature_analysis(X_train: pd.DataFrame, y_train: pd.Series) -> None:
    """Mutual information and VIF before and after feature engineering; writes features.json."""
    frame = FeatureEngineer().fit(X_train).transform(X_train)
    columns = [*NUMERIC, *ENGINEERED]
    values = frame[columns].astype(float)

    # The counts are discrete, and the k-NN estimator used for continuous
    # features breaks down on their massive ties; integer columns get the
    # contingency-table estimator instead.
    discrete = (values == values.round()).all().to_numpy()
    mi = pd.Series(
        mutual_info_classif(
            values,
            y_train,
            discrete_features=discrete,
            n_neighbors=3,
            random_state=SEED,
        ),
        index=columns,
    ).sort_values()
    spearman = values.apply(lambda c: c.corr(y_train, method="spearman"))
    inflation = pd.Series(vif(values))

    # Degree-2 terms of the transformed base counts, the polynomial variant.
    base = PowerTransformer().fit_transform(frame[NUMERIC].astype(float))
    poly = PolynomialFeatures(degree=2, include_bias=False).fit_transform(base)
    poly_vif = np.array(list(vif(pd.DataFrame(poly)).values()))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = ["darkorange" if c in ENGINEERED else "steelblue" for c in mi.index]
    axes[0].barh(mi.index, mi.to_numpy(), color=colors)
    axes[0].set_xlabel("Взаємна інформація з цільовою змінною, нат")
    axes[0].set_title("Інформативність ознак")
    axes[0].tick_params(axis="y", labelsize=8)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="steelblue"),
        plt.Rectangle((0, 0), 1, 1, color="darkorange"),
    ]
    axes[0].legend(handles, ["вихідні", "згенеровані"], fontsize=8, loc="lower right")

    shown = inflation.clip(upper=VIF_DISPLAY_CAP).sort_values()
    colors = ["darkorange" if c in ENGINEERED else "steelblue" for c in shown.index]
    bars = axes[1].barh(shown.index, shown.to_numpy(), color=colors)
    axes[1].bar_label(
        bars,
        labels=[
            "∞" if inflation[c] >= VIF_DISPLAY_CAP else f"{inflation[c]:.1f}"
            for c in shown.index
        ],
        fontsize=7,
    )
    axes[1].set_xscale("log")
    axes[1].axvline(10, color="crimson", linestyle="--", linewidth=1, label="VIF = 10")
    axes[1].set_xlabel("VIF (логарифмічна шкала)")
    axes[1].set_title("Мультиколінеарність після інженерії ознак")
    axes[1].tick_params(axis="y", labelsize=8)
    axes[1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    save_figure(fig, LAB1 / "features.png")

    diag = frame[["diag_1", "diag_1_group"]].astype("string").fillna("Missing")
    save_json(
        {
            "mutual_information": mi.sort_values(ascending=False).round(5).to_dict(),
            "spearman_with_target": spearman.round(4).to_dict(),
            "vif": inflation.to_dict(),
            "poly_n_features": poly.shape[1],
            "poly_vif_median": float(np.median(poly_vif)),
            "poly_vif_max": float(poly_vif.max()),
            "poly_vif_over_10": int((poly_vif > 10).sum()),
            "cramers_v_diag_1": cramers_v(diag["diag_1"], y_train),
            "cramers_v_diag_1_group": cramers_v(diag["diag_1_group"], y_train),
        },
        LAB1 / "features.json",
    )
    log.info("Top MI: %s", mi.sort_values(ascending=False).head(6).round(4).to_dict())
