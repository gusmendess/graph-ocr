"""Figuras de comparação dos dois primeiros estágios da cascata, uma por par e por algoritmo, para slides.

Cada par (consulta A, candidato B) gera duas figuras 1x3, no estilo dos slides do projeto:

    sha256:  [original] A          [original] B          |A - B| original
    phash:   [phash] A pré-proc.   [phash] B pré-proc.   |A - B| phash

O sha256 não tem pré-processamento: ele só diz se os bytes são idênticos, e a diferença dos pixels originais fica
toda preta quando são. O phash olha só a miniatura 32x32 em tons de cinza.
"""

from dataclasses import dataclass

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from app.api.services import hashing_service as hs  # noqa: E402
from app.api.services.vector_visualization_service import _RC, INK, MUTED, INK2, _png  # noqa: E402


@dataclass
class HashCandidate:
    label: str
    sha: str
    image: np.ndarray  # BGR


def _gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _show(ax, data, title: str, cmap=None, vmax: float = 255.0) -> object:
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=11, color=INK2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_anchor("N")  # imagens de proporções diferentes começam todas no topo da célula (títulos alinhados)
    return im


def _heat(fig, ax, diff: np.ndarray, title: str) -> None:
    im = _show(ax, diff, title, cmap="hot")
    # panchor=False: a barra não reposiciona o painel (que continua ancorado no topo, como os outros)
    bar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, ticks=[0, 50, 100, 150, 200, 250],
                       anchor=(0.0, 1.0), panchor=False)
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=9)


def _header(fig, headline: str, note: str) -> None:
    height = fig.get_size_inches()[1]  # margens em polegadas: o espaçamento fica igual em figuras de alturas diferentes
    fig.text(0.01, 1 - 0.12 / height, headline, ha="left", va="top", fontsize=14, fontweight="bold", color=INK)
    fig.text(0.01, 1 - 0.62 / height, note, ha="left", va="top", fontsize=10.5, color=MUTED)


def render_pair_figures(
    query_label: str, query_sha: str, query_image: np.ndarray, candidate: HashCandidate, threshold: float
) -> dict[str, bytes]:
    """{"sha256": png, "phash": png} para o par (A = consulta, B = candidato)."""
    a, b = query_image, candidate.image
    identical = query_sha == candidate.sha
    same_chars = sum(x == y for x, y in zip(query_sha, candidate.sha))

    resized = a.shape[:2] != b.shape[:2]
    b_same = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA) if resized else b
    diff_original = cv2.absdiff(_gray(a), _gray(b_same))

    thumb_a, thumb_b = hs.phash_thumbnail(a), hs.phash_thumbnail(b)
    diff_phash = cv2.absdiff(thumb_a, thumb_b)
    bits_differ = int(np.count_nonzero(hs.phash(a) != hs.phash(b)))
    similarity = 1 - bits_differ / 64
    passes = similarity >= threshold

    out: dict[str, bytes] = {}
    with plt.rc_context(_RC):
        fig = plt.figure(figsize=(14, 8.6))
        gs = fig.add_gridspec(1, 3, wspace=0.12, top=0.86, bottom=0.03)
        _show(fig.add_subplot(gs[0, 0]), _gray(a), f"[original] A: {query_label}", cmap="gray")
        _show(fig.add_subplot(gs[0, 1]), _gray(b), f"[original] B: {candidate.label}", cmap="gray")
        _heat(fig, fig.add_subplot(gs[0, 2]), diff_original,
              "|A − B| original" + (" (B redimensionada para A)" if resized else ""))
        _header(
            fig,
            f"sha256: {'bytes idênticos' if identical else 'arquivos diferentes'} ({same_chars}/64 caracteres do hash iguais)",
            f"A {query_sha[:16]}…   B {candidate.sha[:16]}…   ·   ao acaso, ~4 de 64 caracteres coincidem",
        )
        out["sha256"] = _png(fig)

        fig = plt.figure(figsize=(14, 5.6))
        gs = fig.add_gridspec(1, 3, wspace=0.12, top=0.80, bottom=0.04)
        _show(fig.add_subplot(gs[0, 0]), thumb_a, "[phash] A pré-processado (32×32)", cmap="gray")
        _show(fig.add_subplot(gs[0, 1]), thumb_b, "[phash] B pré-processado (32×32)", cmap="gray")
        _heat(fig, fig.add_subplot(gs[0, 2]), diff_phash, f"|A − B| phash ({bits_differ}/64 bits diferem)")
        _header(
            fig,
            f"phash: similaridade {similarity:.3f} {'≥' if passes else '<'} limiar {threshold:.2f}: "
            + ("passa para o vetor" if passes else "barrado pelo phash"),
            f"A {query_label}   B {candidate.label}   ·   64 bits = coeficientes de baixa frequência da miniatura acima da mediana",
        )
        out["phash"] = _png(fig)
    return out
