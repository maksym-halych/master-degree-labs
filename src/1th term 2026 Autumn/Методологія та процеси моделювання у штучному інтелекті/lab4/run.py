"""Produces every artifact of the lab 4 report: data, hyperparameter search, benchmark.

RUN — from the workspace root. `--project` points uv at the subject directory,
which holds the pyproject.toml pinning this lab's interpreter:

    uv run --project "src/1th term 2026 Autumn/Методологія та процеси моделювання у штучному інтелекті" \
        python "src/1th term 2026 Autumn/Методологія та процеси моделювання у штучному інтелекті/lab4/run.py"

Add `--stage <name>` (repeatable) to rerun only some stages. The benchmark
loads the model the search saved, so the search must have run at least once.
The latencies in the report are wall-clock times of this machine's CPU, so
nothing else heavy should run beside the benchmark.

The dataset is not downloaded by code: MVTec hands out the link only after a
licence form. Place bottle.tar.xz in .cache/datasets/<term>/<subject>/ and the
data stage extracts it there. The pretrained backbones are downloaded on first
use into .cache/models/hf-hub/. Figures, metrics and the trained model go to
.cache/reports-artifacts/<term>/<subject>/lab4/ and the run log to
logs/<term>/<subject>/lab4/ — all computed by mk1.config. This script does not
build the PDF; `make render-report` at the workspace root does.

What lives here is orchestration only: the list of stages, logging, and the check
that every declared artifact really was written by this run. The computation
lives in mk1.
"""

import argparse
import logging
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

from mk1.config import LAB4, WEIGHTS

# huggingface_hub reads its cache directory once, when it is first imported,
# and timm downloads the backbones through it — so the path has to be set
# before the mk1 modules below pull timm in.
os.environ["HF_HUB_CACHE"] = str(WEIGHTS)

from mk1.ae_search import CHECKPOINT, run_search
from mk1.benchmark import run_benchmark
from mk1.logs import configure_logging
from mk1.mvtec import run_data

log = logging.getLogger("lab4")


def main() -> int:
    """Runs the selected stages in order. Returns the process exit code."""
    log_file = configure_logging("lab4")
    log.info("Logging to %s", log_file)
    log.info("Backbone weights cache: %s", WEIGHTS)

    # What each stage is obliged to leave on disk. A silently missing or stale
    # figure would otherwise only surface at the PDF build, or not at all.
    stages: dict[str, tuple[Callable[[], None], tuple[Path, ...]]] = {
        "data": (
            run_data,
            tuple(
                LAB4 / f
                for f in ("data.json", "samples.png", "patches.png", "defect_area.png")
            ),
        ),
        "search": (
            run_search,
            (
                CHECKPOINT,
                *(LAB4 / f for f in ("search.json", "search.png", "training.png")),
            ),
        ),
        "benchmark": (
            run_benchmark,
            tuple(
                LAB4 / f
                for f in ("benchmark.json", "quality_cost.png", "anomaly_maps.png")
            ),
        ),
    }

    parser = argparse.ArgumentParser(description="Produce the lab 4 artifacts.")
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
