"""Writing the figures and metric files the report is built from.

A plot is flat fills and thin lines, that is a few dozen unique colours, yet
matplotlib writes it as full-colour RGB. A 256-colour palette changes nothing
visually but strips most of the PNG's weight, and with it the report PDF's,
which embeds these figures without recompressing them.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib.figure import Figure
from matplotlib.pyplot import close
from PIL import Image

PALETTE_COLORS = 256


def save_figure(fig: Figure, outpath: Path) -> None:
    """
    Save a figure as a palette PNG instead of full RGB, then close it.

    Args:
        fig: The matplotlib figure.
        outpath: Destination path; parent directories are created as needed.
    """
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight", dpi=150)
    close(fig)

    with Image.open(outpath) as saved:
        # Pillow does not carry the resolution (the pHYs chunk) over on write,
        # and graphicx derives the figure's own size from it — without it the
        # PNG is taken to be 72 dpi.
        dpi = saved.info.get("dpi")
        # convert() reads the file to the end, so the overwrite below is safe.
        rgb = saved.convert("RGB")

    # MAXCOVERAGE rather than the faster FASTOCTREE: that one shifts the white
    # background to (254,254,254) and thereby changes every pixel of the figure.
    # No dithering — it scatters flat fills into noise that PNG can no longer
    # compress, leaving a heavier file.
    palette = rgb.quantize(
        colors=PALETTE_COLORS,
        method=Image.Quantize.MAXCOVERAGE,
        dither=Image.Dither.NONE,
    )
    palette.save(outpath, optimize=True, **({"dpi": dpi} if dpi else {}))


def _to_builtin(value: object) -> object:
    """JSON fallback for the numpy scalars and arrays that metrics come back as."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def save_json(data: dict, outpath: Path) -> None:
    """
    Write a stage's numbers as indented UTF-8 JSON.

    Args:
        data: The stage summary; numpy scalars and arrays are converted.
        outpath: Destination path; parent directories are created as needed.
    """
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=_to_builtin),
        encoding="utf-8",
    )
