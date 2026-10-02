"""Saving figures in a compressed form.

A plot is flat fills and thin lines, that is a few dozen unique colours, yet
matplotlib writes it as full-colour RGB. A 256-colour palette changes nothing
visually but strips ~70% off the PNG, and with it off the report PDF these
figures are embedded into without being recompressed.
"""

from pathlib import Path

from matplotlib.figure import Figure
from PIL import Image

PALETTE_COLORS = 256


def save_figure(fig: Figure, outpath: Path) -> None:
    """
    Save a figure as a palette PNG instead of full RGB.

    Args:
        fig: The matplotlib figure; the caller closes it itself.
        outpath: Destination path; parent directories are created as needed.
    """
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight")

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
