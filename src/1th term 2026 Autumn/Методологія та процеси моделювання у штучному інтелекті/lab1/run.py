"""Produces every artifact of the lab 1 report: EDA, preprocessing and augmentation.

RUN — from the workspace root. `--project` points uv at the subject directory,
which holds the pyproject.toml pinning this lab's interpreter:

    uv run --project "src/1th term 2026 Autumn/Методологія та процеси моделювання у штучному інтелекті" \
        python "src/1th term 2026 Autumn/Методологія та процеси моделювання у штучному інтелекті/lab1/run.py"

Add `--stage <name>` (repeatable) to rerun only some stages; the method
comparison ("experiments") takes the bulk of the run time, about half an hour.

Every path is derived from __file__, so the working directory does not matter.
The dataset is downloaded once into .cache/datasets/, figures and metrics go to
.cache/reports-artifacts/<term>/<subject>/lab1/ and the run log to
logs/<term>/<subject>/lab1/ — all computed by mk1.config. This script does not
build the PDF; `make render-report` at the workspace root does.

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

from mk1.augmentation import run_augmentation
from mk1.config import LAB1
from mk1.data import clean, load_raw, split
from mk1.eda import run_eda
from mk1.evaluation import run_experiments, run_final
from mk1.feature_analysis import run_feature_analysis
from mk1.imputation import run_imputation
from mk1.logs import configure_logging
from mk1.outliers import run_outliers
from mk1.scaling import run_scaling

log = logging.getLogger("lab1")


def main() -> int:
    """Runs the selected stages in order. Returns the process exit code."""
    log_file = configure_logging("lab1")
    log.info("Logging to %s", log_file)

    raw = load_raw()
    df, steps = clean(raw)
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
        "eda": (
            lambda: run_eda(raw, df, steps),
            tuple(
                LAB1 / f
                for f in (
                    "class_balance.png",
                    "missingness.png",
                    "numeric_distributions.png",
                    "qq_plots.png",
                    "correlation.png",
                    "cramers_v.png",
                    "cardinality.png",
                    "eda.json",
                )
            ),
        ),
        "outliers": (
            lambda: run_outliers(df),
            tuple(
                LAB1 / f
                for f in (
                    "outliers_summary.png",
                    "outliers_jaccard.png",
                    "outliers_pca.png",
                    "outliers.json",
                )
            ),
        ),
        "imputation": (
            lambda: run_imputation(X_train, y_train),
            tuple(
                LAB1 / f
                for f in (
                    "imputation_numeric.png",
                    "imputation_categorical.png",
                    "imputation.json",
                )
            ),
        ),
        "scaling": (
            lambda: run_scaling(X_train),
            tuple(LAB1 / f for f in ("scaling.png", "scaling.json")),
        ),
        "features": (
            lambda: run_feature_analysis(X_train, y_train),
            tuple(LAB1 / f for f in ("features.png", "features.json")),
        ),
        "augmentation": (
            lambda: run_augmentation(X_train, y_train),
            tuple(
                LAB1 / f
                for f in (
                    "augmentation_counts.png",
                    "augmentation_pca.png",
                    "augmentation_territory.png",
                    "augmentation.json",
                )
            ),
        ),
        "experiments": (
            lambda: run_experiments(X_train, y_train),
            tuple(
                LAB1 / f
                for f in ("experiments.png", "noise_sweep.png", "experiments.json")
            ),
        ),
        "final": (
            lambda: run_final(X_train, X_test, y_train, y_test),
            tuple(LAB1 / f for f in ("final_curves.png", "final.json")),
        ),
    }

    parser = argparse.ArgumentParser(description="Produce the lab 1 artifacts.")
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
