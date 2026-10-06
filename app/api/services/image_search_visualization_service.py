"""Folha de contato da busca de imagens por texto: as top-k fotos, em ordem, com o cosseno de cada uma (para slides)."""

import math

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from app.api.services.vector_visualization_service import _RC, GRID, INK, INK2, MUTED, _png  # noqa: E402

HIGHLIGHT = "#2a78d6"  # primeira cor categórica da paleta validada


def render_contact_sheet(query: str, tiles: list[tuple[str, float, np.ndarray]], n_indexed: int, cols: int | None = None) -> bytes:
    """`tiles`: (rótulo, cosseno, imagem BGR) na ordem do ranking. O 1º colocado leva moldura azul.

    Colunas: até 6 fotos em 3 colunas (fica legível num slide); acima disso, 5.
    """
    cols = cols or (min(len(tiles), 3) if len(tiles) <= 6 else 5)
    rows = max(math.ceil(len(tiles) / cols), 1)
    cols = min(cols, max(len(tiles), 1))
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.1, rows * 3.9 + 1.0), squeeze=False)
        for ax in axes.flat:
            ax.set_axis_off()
        for n, (ax, (label, score, image)) in enumerate(zip(axes.flat, tiles), 1):
            ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            ax.set_title(f"{n}. {label}", fontsize=10.5, color=INK, loc="left", pad=6)
            ax.text(0, -0.02, f"cosseno {score:.3f}", transform=ax.transAxes, va="top", fontsize=10, color=INK2)
            frame = Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False,
                              edgecolor=HIGHLIGHT if n == 1 else GRID, linewidth=3 if n == 1 else 1.2, clip_on=False)
            ax.add_patch(frame)
            ax.set_axis_on()
            ax.set_anchor("N")  # fotos de proporções diferentes começam todas no topo da célula (títulos alinhados)
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ax.spines.values():
                side.set_visible(False)
        fig.suptitle(f'Busca: "{query}"', x=0.01, y=0.995, ha="left", fontsize=17, fontweight="bold", color=INK)
        fig.text(0.01, 0.945 if rows == 1 else 0.96, f"top {len(tiles)} de {n_indexed} imagens indexadas  ·  ordem decrescente de cosseno (SigLIP2)",
                 fontsize=10.5, color=MUTED, va="top")
        fig.subplots_adjust(left=0.02, right=0.98, top=0.84 if rows == 1 else 0.89, bottom=0.04, hspace=0.38, wspace=0.12)
        return _png(fig)
