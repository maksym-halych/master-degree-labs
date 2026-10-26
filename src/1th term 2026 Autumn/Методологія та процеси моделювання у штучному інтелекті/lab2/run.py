"""Produces every artifact of the lab 2 report: classifier comparison across data scenarios.

RUN — from the workspace root. `--project` points uv at the subject directory,
which holds the pyproject.toml pinning this lab's interpreter:

    uv run --project "src/1th term 2026 Autumn/Методологія та процеси моделювання у штучному інтелекті" \
        python "src/1th term 2026 Autumn/Методологія та процеси моделювання у штучному інтелекті/lab2/run.py"

Add `--stage <name>` (repeatable) to rerun only some stages. The test stage
reads the decision thresholds the comparison stage wrote, so it needs that
stage to have run at least once.

Every path is derived from __file__, so the working directory does not matter.
The dataset is the one lab 1 downloaded into .cache/datasets/ (fetched again if
missing), figures and metrics go to .cache/reports-artifacts/<term>/<subject>/lab2/
and the run log to logs/<term>/<subject>/lab2/ — all computed by mk1.config.
This script does not build the PDF; `make render-report` at the workspace root
does.

What lives here is orchestration only: the list of stages, logging, and the check
that every declared artifact really was written by this run. The computation
lives in mk1.
"""

import argparse
import logging
import sys
import time
from collections.abc import Callable
from pathlib import Path

from mk1.comparison import run_comparison, run_test
from mk1.config import LAB2
from mk1.data import clean, load_raw, split
from mk1.logs import configure_logging
from mk1.overfitting import run_overfitting

log = logging.getLogger("lab2")


def main() -> int:
    """Runs the selected stages in order. Returns the process exit code."""
    log_file = configure_logging("lab2")
    log.info("Logging to %s", log_file)

    df, _ = clean(load_raw())
    X_train, X_test, y_train, y_test = split(df)
    log.info(
        "Train %d rows, test %d rows, positive share %.2f%%",
        len(X_train),
        len(X_test),
        y_train.mean() * 100,
    )

    # What each stage is obliged to leave on disk. A silently missing or stale
    # figure would otherwise only surface at the PDF build, or not at all.
    stages: dict[str, tuple[Callable[[], None], tuple[Path, ...]]] = {
        "comparison": (
            lambda: run_comparison(X_train, y_train),
            tuple(
                LAB2 / f
                for f in (
                    "comparison_metrics.png",
                    "overfitting_gap.png",
                    "comparison.json",
                )
            ),
        ),
        "overfitting": (
            lambda: run_overfitting(X_train, y_train),
            tuple(
                LAB2 / f
                for f in (
                    "overfitting_capacity.png",
                    "overfitting_ratio.png",
                    "overfitting.json",
                )
            ),
        ),
        "test": (
            lambda: run_test(X_train, X_test, y_train, y_test),
            tuple(LAB2 / f for f in ("test_curves.png", "test.json")),
        ),
    }

    parser = argparse.ArgumentParser(description="Produce the lab 2 artifacts.")
    parser.add_argument(
        "--stage",
        action="append",
        choices=list(stages),
        help="run only this stage (repeatable)",
    )
    selected = parser.parse_args().stage or list(stages)

    for name in selected:
        run, expected = stages[name]
        log.info("--- %s ---", name)
        started = time.time()
        try:
            run()
        except Exception:
            log.exception("Stage %r failed", name)
            return 1

        stale = [p for p in expected if not p.exists() or p.stat().st_mtime < started]
        if stale:
            for path in stale:
                log.error("Stage %r did not write the artifact: %s", name, path)
            return 1
        log.info(
            "Stage %r done in %.0fs, artifacts verified: %d",
            name,
            time.time() - started,
            len(expected),
        )

    log.info("Done. Build the report with: make render-report")
    return 0


if __name__ == "__main__":
    sys.exit(main())
