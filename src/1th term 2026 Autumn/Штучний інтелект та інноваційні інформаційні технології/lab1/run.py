"""Produces every artifact of the lab 1 report (TensorFlow).

RUN — from the workspace root, with REPORT pointing at this directory:

    make run-lab REPORT="src/1th term 2026 Autumn/Штучний інтелект та інноваційні інформаційні технології/lab1"

With REPORT set in .env, `make run-lab` on its own does the same. Without make:

    uv run --project "src/1th term 2026 Autumn/Штучний інтелект та інноваційні інформаційні технології" \
        python "src/1th term 2026 Autumn/Штучний інтелект та інноваційні інформаційні технології/lab1/run.py"

Every path is derived from __file__, so the working directory does not matter.
Figures and metrics go to .cache/reports-artifacts/<term>/<subject>/ and the run
log to logs/<term>/<subject>/lab1/ — both computed by aiit.config. This script
does not build the PDF; `make render-report` at the workspace root does.

What lives here is orchestration only: the list of stages, logging, and the check
that every declared artifact really did appear. The computation lives in aiit.
"""

import logging
import sys

from aiit.config import EDA, LAB1
from aiit.eda import run_eda
from aiit.logs import configure_logging
from aiit.run_lab1 import run_tensorflow_experiments

log = logging.getLogger("lab1")

# What this script is obliged to leave on disk. Checked after every stage: a
# silently missing figure would otherwise only surface at the PDF build.
STAGES: tuple[tuple[str, object, tuple], ...] = (
    (
        "Exploratory data analysis",
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
        "TensorFlow experiments",
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
    """Runs the stages in order. Returns the process exit code."""
    log_file = configure_logging("lab1")
    log.info("Logging to %s", log_file)

    for title, run, expected in STAGES:
        log.info("--- %s ---", title)
        try:
            run()
        except Exception:
            log.exception("Stage %r failed", title)
            return 1

        missing = [p for p in expected if not p.exists()]
        if missing:
            for path in missing:
                log.error("Stage %r did not produce the artifact: %s", title, path)
            return 1
        log.info("Artifacts verified: %d", len(expected))

    log.info("Done. Build the report with: make render-report")
    return 0


if __name__ == "__main__":
    sys.exit(main())
