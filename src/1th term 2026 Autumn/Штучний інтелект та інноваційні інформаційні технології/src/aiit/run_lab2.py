"""Лабораторна робота №2: експерименти у PyTorch (завдання 6-11).

Повторює експерименти лабораторної роботи №1 засобами PyTorch і додатково
формує порівняння двох фреймворків.
"""

import json
import logging
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from aiit.config import BATCH_SIZE, EPOCHS, LAB1, LAB2, LEARNING_RATE, SEED
from aiit.data import prepare
from aiit.metrics import regression_metrics
from aiit.plots import (
    framework_comparison,
    learning_curves,
    overfit_gap,
    predicted_vs_actual,
)
from aiit.torch_models import all_models

log = logging.getLogger(__name__)

LAB1_METRICS = LAB1 / "lab1.json"


def set_seeds() -> None:
    """Фіксує всі три генератори випадкових чисел перед кожним навчанням."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def make_loader(X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y).unsqueeze(1))
    generator = torch.Generator().manual_seed(SEED) if shuffle else None
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, generator=generator)


@torch.no_grad()
def evaluate(model: nn.Module, X: np.ndarray) -> np.ndarray:
    """Прогноз у режимі eval: Dropout вимкнено, BatchNorm використовує
    накопичені статистики, а не статистики поточного батчу."""
    model.eval()
    return model(torch.from_numpy(X)).numpy()


def train_one(model: nn.Module, weight_decay: float, ds: dict) -> tuple[dict, float]:
    """Ручний цикл навчання — на відміну від Keras, PyTorch не має .fit()."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=weight_decay
    )

    train_loader = make_loader(ds["X_train"], ds["y_train"], shuffle=True)
    X_val = torch.from_numpy(ds["X_val"])
    y_val = torch.from_numpy(ds["y_val"]).unsqueeze(1)

    history: dict[str, list[float]] = {"loss": [], "val_loss": []}
    start = time.perf_counter()

    for _ in range(EPOCHS):
        model.train()
        running, seen = 0.0, 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(xb)
            seen += len(xb)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val).item()

        history["loss"].append(running / seen)
        history["val_loss"].append(val_loss)

    return history, time.perf_counter() - start


def run_pytorch_experiments() -> None:
    """Навчає всі конфігурації, зберігає метрики, рисунки та порівняння.

    Raises:
        FileNotFoundError: Якщо метрик ЛР №1 немає — без них неможливо
            побудувати порівняння фреймворків, яке вимагає звіт ЛР №2.
    """
    if not LAB1_METRICS.exists():
        raise FileNotFoundError(LAB1_METRICS)

    set_seeds()
    torch.set_num_threads(max(1, torch.get_num_threads()))

    ds = prepare()
    input_dim = ds["X_train"].shape[1]
    log.info(
        "Ознак: %d; train=%d, val=%d, test=%d",
        input_dim,
        len(ds["X_train"]),
        len(ds["X_val"]),
        len(ds["X_test"]),
    )

    models = all_models(input_dim)
    results: dict[str, dict] = {}
    histories: dict[str, dict] = {}
    trained: dict[str, nn.Module] = {}

    for name, (model, weight_decay) in models.items():
        log.info("PyTorch: %s", name)
        set_seeds()

        history, elapsed = train_one(model, weight_decay, ds)
        trained[name] = model

        results[name] = {
            "val": regression_metrics(ds["y_val"], evaluate(model, ds["X_val"])),
            "test": regression_metrics(ds["y_test"], evaluate(model, ds["X_test"])),
            "train_time_s": round(elapsed, 2),
            "n_params": sum(p.numel() for p in model.parameters()),
            "final_train_loss": history["loss"][-1],
            "final_val_loss": history["val_loss"][-1],
        }
        histories[name] = history

        r = results[name]
        log.info(
            "RMSE(test)=%.4f  R2(test)=%.4f  час=%.1f с  параметрів=%d",
            r["test"]["rmse"],
            r["test"]["r2"],
            elapsed,
            r["n_params"],
        )

    learning_curves(histories, LAB2 / "learning_curves.png", "PyTorch: навчальні криві")
    overfit_gap(
        histories,
        LAB2 / "overfit_gap.png",
        "PyTorch: розрив між валідаційною та тренувальною втратою",
    )

    best = min(results, key=lambda k: results[k]["test"]["rmse"])
    predicted_vs_actual(
        ds["y_test"],
        evaluate(trained[best], ds["X_test"]),
        LAB2 / "pred_vs_actual.png",
        f"PyTorch, найкраща модель: {best}",
    )

    # Порівняння фреймворків (завдання 11) — рисунок обов'язковий для звіту,
    # тому відсутність метрик ЛР №1 зупиняє запуск ще на початку функції.
    tf_results = json.loads(LAB1_METRICS.read_text(encoding="utf-8"))["results"]
    framework_comparison(tf_results, results, LAB2 / "framework_comparison.png")

    LAB2.mkdir(parents=True, exist_ok=True)
    (LAB2 / "lab2.json").write_text(
        json.dumps(
            {
                "framework": "pytorch",
                "results": results,
                "histories": histories,
                "best": best,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info("Найкраща модель: %s", best)
