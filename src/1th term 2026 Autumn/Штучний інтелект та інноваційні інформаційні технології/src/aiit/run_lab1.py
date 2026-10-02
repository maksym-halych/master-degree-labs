"""Лабораторна робота №1: експерименти у TensorFlow (завдання 6-10).

Навчає чотири конфігурації моделей, зберігає метрики lab1.json та рисунки
у теці артефактів (config.LAB1).
"""

import json
import logging
import random
import time

import numpy as np

from aiit.config import BATCH_SIZE, EPOCHS, LAB1, SEED
from aiit.data import prepare
from aiit.metrics import regression_metrics
from aiit.plots import learning_curves, overfit_gap, predicted_vs_actual
from aiit.tf_models import all_models, set_seed

log = logging.getLogger(__name__)


def set_seeds() -> None:
    """Фіксує всі три генератори випадкових чисел перед кожним навчанням."""
    random.seed(SEED)
    np.random.seed(SEED)
    set_seed(SEED)


def run_tensorflow_experiments() -> None:
    """Навчає всі конфігурації, зберігає метрики та рисунки ЛР №1."""
    set_seeds()

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

    for name, model in models.items():
        log.info("TensorFlow: %s", name)
        set_seeds()

        start = time.perf_counter()
        hist = model.fit(
            ds["X_train"],
            ds["y_train"],
            validation_data=(ds["X_val"], ds["y_val"]),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=0,
        )
        elapsed = time.perf_counter() - start

        pred_val = model.predict(ds["X_val"], verbose=0)
        pred_test = model.predict(ds["X_test"], verbose=0)

        results[name] = {
            "val": regression_metrics(ds["y_val"], pred_val),
            "test": regression_metrics(ds["y_test"], pred_test),
            "train_time_s": round(elapsed, 2),
            "n_params": int(model.count_params()),
            "final_train_loss": float(hist.history["loss"][-1]),
            "final_val_loss": float(hist.history["val_loss"][-1]),
        }
        histories[name] = {
            "loss": [float(v) for v in hist.history["loss"]],
            "val_loss": [float(v) for v in hist.history["val_loss"]],
        }

        r = results[name]
        log.info(
            "RMSE(test)=%.4f  R2(test)=%.4f  час=%.1f с  параметрів=%d",
            r["test"]["rmse"],
            r["test"]["r2"],
            elapsed,
            r["n_params"],
        )

    learning_curves(
        histories, LAB1 / "learning_curves.png", "TensorFlow: навчальні криві"
    )
    overfit_gap(
        histories,
        LAB1 / "overfit_gap.png",
        "TensorFlow: розрив між валідаційною та тренувальною втратою",
    )

    best = min(results, key=lambda k: results[k]["test"]["rmse"])
    best_pred = models[best].predict(ds["X_test"], verbose=0)
    predicted_vs_actual(
        ds["y_test"],
        best_pred,
        LAB1 / "pred_vs_actual.png",
        f"TensorFlow, найкраща модель: {best}",
    )

    LAB1.mkdir(parents=True, exist_ok=True)
    (LAB1 / "lab1.json").write_text(
        json.dumps(
            {
                "framework": "tensorflow",
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
