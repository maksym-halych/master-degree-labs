"""Готує всі артефакти звіту ЛР №2 (PyTorch).

ЗАПУСК — з теки дисципліни (там, де лежить pyproject.toml):

    uv run python lab2/run.py

ПОТРЕБУЄ ЛР №1: рисунок порівняння фреймворків будується з метрик lab1.json,
тому спершу треба виконати `uv run python lab1/run.py`. Скрипт перевіряє це
до початку навчання, а не після.

Рисунки й метрики пишуться у .cache/reports-artifacts/<семестр>/<дисципліна>/,
шлях обчислює aiit.config. Сам PDF цей скрипт не збирає — для цього є
`make render-report` у корені робочого простору.
"""

import logging
import sys

from aiit.config import LAB2
from aiit.run_lab2 import LAB1_METRICS, run_pytorch_experiments

log = logging.getLogger("lab2")

# Що цей скрипт зобов'язаний залишити на диску. Перевіряється після етапу:
# мовчазно відсутній рисунок інакше виявиться аж на збірці PDF.
STAGES: tuple[tuple[str, object, tuple], ...] = (
    (
        "Експерименти PyTorch і порівняння фреймворків",
        run_pytorch_experiments,
        (
            LAB2 / "learning_curves.png",
            LAB2 / "overfit_gap.png",
            LAB2 / "pred_vs_actual.png",
            LAB2 / "framework_comparison.png",
            LAB2 / "lab2.json",
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

    # Перевіряємо до навчання: інакше про відсутність метрик ЛР №1 стало б
    # відомо аж наприкінці, після кількох хвилин роботи.
    if not LAB1_METRICS.exists():
        log.error("Немає метрик ЛР №1: %s", LAB1_METRICS)
        log.error("Спершу виконайте: uv run python lab1/run.py")
        return 1

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
