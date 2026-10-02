"""Готує всі артефакти звіту ЛР №1 (TensorFlow).

ЗАПУСК — з теки дисципліни (там, де лежить pyproject.toml):

    uv run python lab1/run.py

Рисунки й метрики пишуться у .cache/reports-artifacts/<семестр>/<дисципліна>/,
шлях обчислює aiit.config. Сам PDF цей скрипт не збирає — для цього є
`make render-report` у корені робочого простору.

Тут лише оркестрація: перелік етапів, журналювання та перевірка, що кожен
оголошений артефакт справді з'явився. Обчислення живуть у пакеті aiit.
"""

import logging
import sys

from aiit.config import EDA, LAB1
from aiit.eda import run_eda
from aiit.run_lab1 import run_tensorflow_experiments

log = logging.getLogger("lab1")

# Що цей скрипт зобов'язаний залишити на диску. Перевіряється після кожного
# етапу: мовчазно відсутній рисунок інакше виявиться аж на збірці PDF.
STAGES: tuple[tuple[str, object, tuple], ...] = (
    (
        "Розвідувальний аналіз даних",
        run_eda,
        (
            EDA / "distributions.png",
            EDA / "correlation.png",
            EDA / "target_imbalance.png",
            EDA / "pca_tsne.png",
            EDA / "eda.json",
        ),
    ),
    (
        "Експерименти TensorFlow",
        run_tensorflow_experiments,
        (
            LAB1 / "learning_curves.png",
            LAB1 / "overfit_gap.png",
            LAB1 / "pred_vs_actual.png",
            LAB1 / "lab1.json",
        ),
    ),
)


def main() -> int:
    """Виконує етапи по черзі. Повертає код завершення процесу."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    for title, run, expected in STAGES:
        log.info("--- %s ---", title)
        try:
            run()
        except Exception:
            log.exception("Етап «%s» завершився помилкою", title)
            return 1

        missing = [p for p in expected if not p.exists()]
        if missing:
            for path in missing:
                log.error("Етап «%s» не створив артефакт: %s", title, path)
            return 1
        log.info("Артефактів перевірено: %d", len(expected))

    log.info("Готово. Збірка звіту: make render-report")
    return 0


if __name__ == "__main__":
    sys.exit(main())
