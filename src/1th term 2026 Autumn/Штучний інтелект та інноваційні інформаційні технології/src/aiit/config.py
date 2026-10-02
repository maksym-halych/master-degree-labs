"""Shared hyperparameters for both labs.

Both frameworks import these exact values, so any difference between the
TensorFlow and the PyTorch results comes from the framework itself and not
from a mismatch in the experiment setup.
"""

from pathlib import Path

SEED = 42

# Sample split: 60 / 20 / 20
TEST_SIZE = 0.20
VAL_SIZE = 0.25  # 0.25 of the remaining 80% yields 20% of the full dataset

# Training (task 8: 10-20 epochs, fixed optimizer and learning rate)
EPOCHS = 20
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
L2_LAMBDA = 1e-4
DROPOUT_RATE = 0.2

# Architectures (task 6)
HIDDEN_UNITS = (64, 32)

# Activations to compare (task 7)
ACTIVATIONS = ("relu", "gelu")

# Number of quantile groups of the target used for the imbalance analysis
TARGET_BINS = 5

# src/aiit/config.py -> src/aiit -> src -> the subject directory
ROOT = Path(__file__).resolve().parent.parent.parent
REPO_ROOT = ROOT.parent.parent.parent  # the workspace root

# Where this subject sits under src/, e.g. "1th term 2026 Autumn/<subject>".
# Both the artifact tree and the log tree mirror it, so the term and subject
# names follow the actual directory layout instead of being duplicated here.
SUBJECT_PATH = ROOT.relative_to(REPO_ROOT / "src")

# Every generated artifact (figures, metrics) lands in .cache/ — the workspace's
# shared, untracked tree of intermediate data. The source tree stays clean.
ARTIFACTS = REPO_ROOT / ".cache" / "reports-artifacts" / SUBJECT_PATH

# Per-run log files, one directory per lab, in the workspace's shared logs/ tree.
LOGS = REPO_ROOT / "logs" / SUBJECT_PATH

EDA = ARTIFACTS / "eda"
LAB1 = ARTIFACTS / "lab1"
LAB2 = ARTIFACTS / "lab2"
