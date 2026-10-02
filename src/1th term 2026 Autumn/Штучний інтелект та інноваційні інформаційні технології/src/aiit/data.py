"""Loading and preprocessing of the California Housing dataset.

Tasks 4-5: preprocessing (StandardScaler + PCA) and the train/val/test split.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from aiit.config import SEED, TARGET_BINS, TEST_SIZE, VAL_SIZE


def load_raw() -> tuple[pd.DataFrame, pd.Series]:
    """Returns the features and the target as pandas objects."""
    bunch = fetch_california_housing(as_frame=True)
    return bunch.data, bunch.target


def target_bins(y: pd.Series, n_bins: int = TARGET_BINS) -> pd.Series:
    """Quantile groups of the target.

    The task is a regression, so there is no class imbalance in the literal
    sense. Splitting into quantile groups is what makes it possible to analyse
    the skew of the target distribution (task 3) and to stratify the split.
    """
    return pd.qcut(y, q=n_bins, labels=False, duplicates="drop")


def split(X: pd.DataFrame, y: pd.Series):
    """A 60/20/20 split stratified by the quantile groups of the target.

    Stratification is needed because the distribution of MedHouseVal is
    noticeably skewed and artificially capped at its maximum: a random split
    would leave the samples with differing shares of expensive houses.
    """
    strata = target_bins(y)

    X_tmp, X_test, y_tmp, y_test, s_tmp, _ = train_test_split(
        X,
        y,
        strata,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=strata,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp,
        y_tmp,
        test_size=VAL_SIZE,
        random_state=SEED,
        stratify=s_tmp,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def prepare(use_pca: bool = False, n_components: int = 6):
    """The full data preparation pipeline.

    The scaler is fitted on the training sample only and then applied to the
    validation and test samples — otherwise the test statistics would leak
    into training.
    """
    X, y = load_raw()
    X_train, X_val, X_test, y_train, y_val, y_test = split(X, y)

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    if use_pca:
        pca = PCA(n_components=n_components, random_state=SEED).fit(X_train_s)
        X_train_s = pca.transform(X_train_s)
        X_val_s = pca.transform(X_val_s)
        X_test_s = pca.transform(X_test_s)

    return {
        "X_train": X_train_s.astype(np.float32),
        "X_val": X_val_s.astype(np.float32),
        "X_test": X_test_s.astype(np.float32),
        "y_train": y_train.to_numpy().astype(np.float32),
        "y_val": y_val.to_numpy().astype(np.float32),
        "y_test": y_test.to_numpy().astype(np.float32),
        "feature_names": list(X.columns),
    }
