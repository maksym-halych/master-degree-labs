"""Paths and experiment constants shared by the labs of this subject."""

from pathlib import Path

SEED = 42

# Share of the cleaned dataset held out for the single final evaluation. Every
# method comparison runs cross-validation on the remaining part only.
TEST_SIZE = 0.20
CV_FOLDS = 5

# Shapiro-Wilk p-values are unreliable above 5000 observations (scipy warns),
# so the test runs on a random subsample of this size.
SHAPIRO_SAMPLE = 5000

# KNNImputer and the masking experiment are quadratic in the number of rows;
# a stratified subsample keeps them to seconds without changing the conclusion.
IMPUTATION_SAMPLE = 10_000

# src/mk1/config.py -> src/mk1 -> src -> the subject directory
ROOT = Path(__file__).resolve().parent.parent.parent
REPO_ROOT = ROOT.parent.parent.parent  # the workspace root

# Where this subject sits under src/, e.g. "1th term 2026 Autumn/<subject>".
# The artifact, log and dataset trees all mirror it, so the term and subject
# names follow the actual directory layout instead of being duplicated here.
SUBJECT_PATH = ROOT.relative_to(REPO_ROOT / "src")

# Every generated artifact (figures, metrics) lands in .cache/ — the workspace's
# shared, untracked tree of intermediate data. The source tree stays clean.
ARTIFACTS = REPO_ROOT / ".cache" / "reports-artifacts" / SUBJECT_PATH

# Downloaded datasets, kept apart from the artifacts so that wiping the
# artifacts to force a clean re-run does not force a re-download.
DATASETS = REPO_ROOT / ".cache" / "datasets" / SUBJECT_PATH

# Per-run log files, one directory per lab, in the workspace's shared logs/ tree.
LOGS = REPO_ROOT / "logs" / SUBJECT_PATH

LAB1 = ARTIFACTS / "lab1"
