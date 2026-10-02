"""Produces every artifact of the lab 2 report (PyTorch).

RUN — from the workspace root. `--project` points uv at the subject directory,
which holds the pyproject.toml pinning this lab's interpreter:

    uv run --project "src/1th term 2026 Autumn/Штучний інтелект та інноваційні інформаційні технології" \
        python "src/1th term 2026 Autumn/Штучний інтелект та інноваційні інформаційні технології/lab2/run.py"

REQUIRES LAB 1: the framework comparison figure is built from the lab1.json
metrics, so lab1/run.py has to run first. This script checks that before
training starts rather than after.

Every path is derived from __file__, so the working directory does not matter.
Figures and metrics go to .cache/reports-artifacts/<term>/<subject>/ and the run
log to logs/<term>/<subject>/lab2/ — both computed by aiit.config. This script
does not build the PDF; `make render-report` at the workspace root does.
"""

import logging
import sys

from aiit.config import LAB2
from aiit.logs import configure_logging
from aiit.run_lab2 import LAB1_METRICS, run_pytorch_experiments

log = logging.getLogger("lab2")

# What this script is obliged to leave on disk. Checked after the stage: a
# silently missing figure would otherwise only surface at the PDF build.
STAGES: tuple[tuple[str, object, tuple], ...] = (
    (
        "PyTorch experiments and framework comparison",
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
    """Runs the stages in order. Returns the process exit code."""
    log_file = configure_logging("lab2")
    log.info("Logging to %s", log_file)

    # Checked before training: otherwise the missing lab 1 metrics would only
    # become known at the very end, after several minutes of work.
    if not LAB1_METRICS.exists():
        log.error("No lab 1 metrics at: %s", LAB1_METRICS)
        log.error("Run lab 1 first: its run.py, from the workspace root")
        return 1

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
