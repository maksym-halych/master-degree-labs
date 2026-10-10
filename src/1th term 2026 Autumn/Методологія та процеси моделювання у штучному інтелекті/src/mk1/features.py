"""Feature engineering: domain features and aggregates by category (task 4).

Everything that learns from data — the group means behind the aggregate
features — is fitted inside a scikit-learn transformer, so cross-validation
recomputes it on each training fold and the held-out fold never leaks in.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from mk1.data import ALL_MEDS

MISSING = "Missing"

# ICD-9 chapters as grouped by the dataset's original study (Strack et al.,
# 2014): roughly 700 codes per diagnosis column fold into nine groups.
ICD9_GROUPS = (
    "diabetes",
    "circulatory",
    "respiratory",
    "digestive",
    "injury",
    "musculoskeletal",
    "genitourinary",
    "neoplasms",
    "other",
)

DIAG_GROUPS = ["diag_1_group", "diag_2_group", "diag_3_group"]

DOMAIN_NUMERIC = [
    "total_visits",
    "meds_per_day",
    "labs_per_day",
    "n_meds_used",
    "n_meds_changed",
]

# (group key, value): the feature is the value's deviation from its group mean.
AGGREGATES = (
    ("medical_specialty", "num_medications"),
    ("medical_specialty", "time_in_hospital"),
    ("diag_1_group", "num_medications"),
    ("diag_1_group", "time_in_hospital"),
)
AGGREGATE_NUMERIC = [f"{value}_vs_{key}" for key, value in AGGREGATES]


def icd9_group(codes: pd.Series) -> pd.Series:
    """
    Map ICD-9 diagnosis codes onto their chapter.

    Args:
        codes: Raw codes such as "250.83", "414" or "V57"; NaN where missing.

    Returns:
        One of ICD9_GROUPS per code, or MISSING.
    """
    text = codes.astype("string")
    # V and E codes (supplementary classifications) are not numeric and fall
    # through to "other".
    n = pd.to_numeric(text, errors="coerce")
    whole = np.floor(n)
    choices = [
        text.str.startswith("250"),
        n.between(390, 459.99) | (whole == 785),
        n.between(460, 519.99) | (whole == 786),
        n.between(520, 579.99) | (whole == 787),
        n.between(800, 999.99),
        n.between(710, 739.99),
        n.between(580, 629.99) | (whole == 788),
        n.between(140, 239.99),
    ]
    # A missing code makes every comparison NA; it is reported as MISSING below.
    masks = [c.to_numpy(dtype=bool, na_value=False) for c in choices]
    groups = np.select(masks, ICD9_GROUPS[:-1], "other")
    return pd.Series(groups, index=codes.index).where(codes.notna(), MISSING)


def add_diag_groups(X: pd.DataFrame) -> pd.DataFrame:
    """Copy of X with the three diagnosis columns mapped onto ICD-9 chapters."""
    out = X.copy()
    for i in (1, 2, 3):
        out[f"diag_{i}_group"] = icd9_group(X[f"diag_{i}"])
    return out


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Adds domain features and deviations from category means.

    Args:
        domain: Add the stateless domain features (visit totals, per-day
            rates, medication counts, ICD-9 chapters).
        aggregates: Add each value's deviation from its group mean, the
            means being learned in fit().
    """

    def __init__(self, domain: bool = True, aggregates: bool = True) -> None:
        self.domain = domain
        self.aggregates = aggregates

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FeatureEngineer":
        """Learn the group means of the aggregate features on the training rows."""
        grouped = add_diag_groups(X)
        self.group_means_ = {}
        self.global_means_ = {}
        for key, value in AGGREGATES:
            keys = grouped[key].fillna(MISSING)
            self.group_means_[(key, value)] = grouped[value].groupby(keys).mean()
            self.global_means_[(key, value)] = grouped[value].mean()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return X with the configured engineered columns appended."""
        out = add_diag_groups(X)
        if self.domain:
            meds = out[ALL_MEDS]
            out["total_visits"] = (
                out["number_outpatient"]
                + out["number_emergency"]
                + out["number_inpatient"]
            )
            # time_in_hospital is at least one day, so the rates are finite.
            out["meds_per_day"] = out["num_medications"] / out["time_in_hospital"]
            out["labs_per_day"] = out["num_lab_procedures"] / out["time_in_hospital"]
            out["n_meds_used"] = (meds != "No").sum(axis=1)
            out["n_meds_changed"] = meds.isin(["Up", "Down"]).sum(axis=1)
        if self.aggregates:
            for key, value in AGGREGATES:
                keys = out[key].fillna(MISSING)
                # A category unseen in training falls back to the global mean,
                # which makes its deviation the plain centred value.
                means = keys.map(self.group_means_[(key, value)]).astype(float)
                means = means.fillna(self.global_means_[(key, value)])
                out[f"{value}_vs_{key}"] = out[value] - means
        return out
