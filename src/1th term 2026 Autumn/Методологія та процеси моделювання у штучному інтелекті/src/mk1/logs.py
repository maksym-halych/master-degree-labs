"""Per-run log files for the lab entry points."""

import logging
from datetime import datetime
from pathlib import Path

from mk1.config import LOGS


def configure_logging(lab: str) -> Path:
    """
    Send logs to the console and to a per-run file under logs/.

    The file always records DEBUG while the console stays at INFO: the method
    comparisons run for minutes, and the detail needed to diagnose a failed run
    has to survive a run nobody was watching.

    Args:
        lab: Name of the lab, used as the log subdirectory (for example "lab1").

    Returns:
        Path to the log file that was opened.

    Raises:
        OSError: If the log directory or file cannot be created.
    """
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    log_dir = LOGS / lab
    log_dir.mkdir(parents=True, exist_ok=True)
    # Local time, matching the timestamps inside the file, so the name of a log
    # lines up with the run you remember starting.
    started = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f"{started}.log"
    to_file = logging.FileHandler(log_file, encoding="utf-8")
    to_file.setLevel(logging.DEBUG)
    to_file.setFormatter(formatter)

    # force=True because basicConfig is a no-op once anything else has touched
    # the root logger, and matplotlib does so on import — getting there first
    # would silently cost us the log file.
    logging.basicConfig(level=logging.DEBUG, handlers=[console, to_file], force=True)
    # Matplotlib logs every font lookup at DEBUG and its INFO records are noise
    # too (the categorical-units fallback, once per bar chart drawn on purpose).
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    return log_file
