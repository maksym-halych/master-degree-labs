"""Збереження рисунків у стисненому вигляді.

Графік — це суцільні заливки й тонкі лінії, тобто кілька десятків унікальних
кольорів, а matplotlib пише їх як повноколірний RGB. Палітра на 256 кольорів
візуально нічого не змінює, але прибирає ~70% ваги PNG, а з нею й ваги PDF
звіту, куди ці рисунки потрапляють без повторного стиснення.
"""

from pathlib import Path

from matplotlib.figure import Figure
from PIL import Image

PALETTE_COLORS = 256


def save_figure(fig: Figure, outpath: Path) -> None:
    """
    Зберегти рисунок як PNG з палітрою замість повного RGB.

    Args:
        fig: Рисунок matplotlib; викликач закриває його сам.
        outpath: Шлях до файлу; батьківські теки створюються за потреби.
    """
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight")

    with Image.open(outpath) as saved:
        # Роздільність (чанк pHYs) Pillow при записі не переносить, а graphicx
        # бере з неї власний розмір рисунка — без неї PNG вважається 72 dpi.
        dpi = saved.info.get("dpi")
        # convert() дочитує файл повністю, тож перезапис нижче безпечний.
        rgb = saved.convert("RGB")

    # MAXCOVERAGE, а не швидший FASTOCTREE: той зсуває білий фон на (254,254,254)
    # і тим змінює кожен піксель рисунка. Без дизерингу — він розсіює рівні
    # заливки на шум, який PNG уже не стискає, і файл виходить важчим.
    palette = rgb.quantize(
        colors=PALETTE_COLORS,
        method=Image.Quantize.MAXCOVERAGE,
        dither=Image.Dither.NONE,
    )
    palette.save(outpath, optimize=True, **({"dpi": dpi} if dpi else {}))
