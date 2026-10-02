"""Завантаження та попередня обробка набору даних California Housing.

Завдання 4-5: препроцесинг (StandardScaler + PCA) і поділ train/val/test.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from aiit.config import SEED, TARGET_BINS, TEST_SIZE, VAL_SIZE


def load_raw() -> tuple[pd.DataFrame, pd.Series]:
    """Повертає ознаки та цільову змінну у вигляді pandas-об'єктів."""
    bunch = fetch_california_housing(as_frame=True)
    return bunch.data, bunch.target


def target_bins(y: pd.Series, n_bins: int = TARGET_BINS) -> pd.Series:
    """Квантильні групи цільової змінної.

    Задача регресійна, тому класового дисбалансу в прямому сенсі немає.
    Розбиття на квантильні групи дозволяє проаналізувати перекіс розподілу
    цільової змінної (завдання 3) і виконати стратифікований поділ вибірки.
    """
    return pd.qcut(y, q=n_bins, labels=False, duplicates="drop")


def split(X: pd.DataFrame, y: pd.Series):
    """Стратифікований за квантилями цільової змінної поділ 60/20/20.

    Стратифікація потрібна тому, що розподіл MedHouseVal має помітний перекіс
    і штучний «стеля»-ефект на максимумі: випадковий поділ дав би вибірки
    з різною часткою дорогих будинків.
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
    """Повний пайплайн підготовки даних.

    Scaler навчається виключно на тренувальній вибірці, після чого
    застосовується до валідаційної та тестової — інакше статистики test
    просочилися б у навчання (data leakage).
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
