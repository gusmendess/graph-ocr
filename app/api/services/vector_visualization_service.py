"""Figuras para slides que comparam os vetores multimodais (visual + texto + layout).

Quatro figuras, cada uma responde uma pergunta, e uma versão combinada:
  bars    - quanto a consulta se parece com cada documento, e qual bloco explica o score?
  matrix  - como todos os documentos se parecem entre si? (grupos aparecem como blocos escuros)
  map     - mapa 2D dos documentos (PCA do vetor composto): quem fica perto de quem?
  map3d   - o mesmo mapa com a 3ª componente do PCA (o que se sobrepõe em 2D pode se separar em 3D).

Cores: os três blocos usam as 3 primeiras categorias da paleta validada (azul, laranja, verde-água), sempre
na mesma ordem; o score composto usa a tinta principal. Magnitude (matriz, mapa, grades) usa uma só
rampa de azul, do claro ao escuro.
"""

import io
import math
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from app.api.domain.vectors import BLOCKS, DocumentVector, normalize_weights  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, BASELINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLOCK_COLOR = {"visual": "#2a78d6", "text": "#eb6834", "layout": "#1baf7a"}
BLOCK_LABEL = {"visual": "visual (aparência)", "text": "texto (conteúdo)", "layout": "layout (posição)"}
# rampa sequencial azul, passos 100 -> 700 da paleta de referência
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#104281", "#0d366b"])
_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
    "font.size": 12,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK2,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "text.color": INK,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
}


@dataclass
class VectorComparison:
    query_label: str
    labels: list[str]  # os documentos do índice + a consulta (sempre uma entrada própria, com o nome que você consultou)
    vectors: list[DocumentVector]
    query_pos: int  # posição da consulta em `labels`
    weights: dict[str, float]  # normalizados sobre os blocos usados
    blocks: list[str]
    identical_pos: int | None = None  # documento do índice com os MESMOS bytes da consulta (sha256 idêntico), se houver

    @property
    def n(self) -> int:
        return len(self.labels)


def build_comparison(query_label, query_sha, query_vector, index, weights) -> VectorComparison:
    labels = [m.get("label", i) for i, m in zip(index.ids, index.metas)]
    vectors = [DocumentVector({b: index.matrices[b][i] for b in index.matrices}) for i in range(len(index))]
    # a consulta é sempre uma entrada própria; se há um documento idêntico no índice, ele continua existindo à parte
    identical = index.ids.index(query_sha) if query_sha in index.ids else None
    labels.append(f"{query_label} (consulta)")
    vectors.append(query_vector)
    blocks = [b for b in BLOCKS if b in query_vector.blocks and b in index.matrices]
    return VectorComparison(
        query_label, labels, vectors, len(labels) - 1, normalize_weights(weights, set(blocks)), blocks, identical
    )


def block_cosine(a: DocumentVector, b: DocumentVector, block: str) -> float:
    return float(np.clip(a.blocks[block] @ b.blocks[block], 0.0, 1.0))  # mesmo truncamento do índice


def composite(cmp: VectorComparison, a: DocumentVector, b: DocumentVector) -> float:
    return sum(w * block_cosine(a, b, blk) for blk, w in cmp.weights.items())


def similarity_matrix(cmp: VectorComparison) -> np.ndarray:
    return np.array([[composite(cmp, a, b) for b in cmp.vectors] for a in cmp.vectors])


def _head(cmp: VectorComparison) -> list[int]:
    """A consulta e, logo abaixo, o documento idêntico a ela (se houver)."""
    return [cmp.query_pos] + ([cmp.identical_pos] if cmp.identical_pos is not None else [])


def _others_by_score(cmp: VectorComparison) -> list[int]:
    """Os demais documentos, do mais ao menos parecido com a consulta (sem a consulta e sem o idêntico a ela)."""
    q = cmp.vectors[cmp.query_pos]
    others = [i for i in range(cmp.n) if i not in _head(cmp)]
    return sorted(others, key=lambda i: -composite(cmp, q, cmp.vectors[i]))


def _query_note(cmp: VectorComparison) -> str:
    text = cmp.labels[cmp.query_pos]
    if cmp.identical_pos is not None:
        text += f"\n= {cmp.labels[cmp.identical_pos]} (idêntico)"
    return text


def _project(cmp: VectorComparison, k: int, pts: list[int] | None = None):
    """PCA do vetor composto. O idêntico à consulta fica de fora: cairia exatamente no mesmo ponto e pesaria em dobro.
    Com `pts`, o PCA é recalculado só com esse subconjunto (a escala passa a ser a das diferenças entre eles)."""
    pts = pts if pts is not None else [i for i in range(cmp.n) if i != cmp.identical_pos]
    x = np.stack([cmp.vectors[i].composite(cmp.weights) for i in pts])
    u, sv, _ = np.linalg.svd(x - x.mean(axis=0), full_matrices=False)
    return pts, {i: r for r, i in enumerate(pts)}, u[:, :k] * sv[:k], (sv**2) / (sv**2).sum()


def _clean(ax, grid_axis: str | None = None) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)


# ---- painéis -----------------------------------------------------------------------------------------------

def draw_bars(ax, cmp: VectorComparison, top: int = 8) -> None:
    q = cmp.vectors[cmp.query_pos]
    # a consulta abre o gráfico como referência (contra ela mesma: 1.000), o documento idêntico logo abaixo
    # (ligado por uma seta), depois os demais do mais ao menos parecido
    docs = _head(cmp) + _others_by_score(cmp)[:top]
    series = [("composite", INK)] + [(b, BLOCK_COLOR[b]) for b in cmp.blocks]
    height, gap = 0.17, 0.02
    group = len(series) * (height + gap)
    for g, i in enumerate(docs):
        base = g * (group + 0.35)
        for s, (name, color) in enumerate(series):
            value = composite(cmp, q, cmp.vectors[i]) if name == "composite" else block_cosine(q, cmp.vectors[i], name)
            y = base + s * (height + gap)
            ax.barh(y, value, height=height, color=color, zorder=3)
            if name == "composite":
                ax.text(value + 0.012, y, f"{value:.3f}", va="center", ha="left", fontsize=11, color=INK, fontweight="bold")
    ax.set_yticks([g * (group + 0.35) + group / 2 - (height + gap) / 2 for g in range(len(docs))])
    names = [cmp.labels[i] if cmp.labels[i].endswith("(consulta)") or i != cmp.query_pos else f"{cmp.labels[i]} (consulta)" for i in docs]
    ax.set_yticklabels(names)
    ax.get_yticklabels()[0].set_fontweight("bold")  # a consulta
    ax.get_yticklabels()[0].set_color(INK)
    if cmp.identical_pos is not None:  # seta entre a consulta e o documento de mesmos bytes
        centers = [g * (group + 0.35) + group / 2 - (height + gap) / 2 for g in (0, 1)]
        ax.annotate("", xy=(1.13, centers[1]), xytext=(1.13, centers[0]), annotation_clip=False,
                    arrowprops={"arrowstyle": "-|>", "color": INK, "linewidth": 1.8, "shrinkA": 0, "shrinkB": 0})
        ax.text(1.16, sum(centers) / 2, "idêntico\n(mesmos bytes,\nsha256)", va="center", ha="left", fontsize=10.5, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.45 if cmp.identical_pos is not None else 1.12)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("similaridade (cosseno, 0 a 1)")
    _clean(ax, "x")
    ax.legend(
        handles=[Patch(color=INK, label="score composto")] + [Patch(color=BLOCK_COLOR[b], label=BLOCK_LABEL[b]) for b in cmp.blocks],
        loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(series), frameon=False, fontsize=11, handlelength=1.2,
    )
    ax.set_title("Score de cada par e o que o explica", loc="left", fontsize=15, fontweight="bold", pad=34)


def draw_matrix(ax, cmp: VectorComparison, fig=None) -> None:
    # consulta primeiro, depois do mais ao menos parecido: os grupos viram blocos escuros contíguos
    order = _head(cmp) + _others_by_score(cmp)
    s = similarity_matrix(cmp)[np.ix_(order, order)]
    off = s[~np.eye(cmp.n, dtype=bool)]
    vmin = max(0.0, math.floor(float(off.min()) * 20) / 20) if off.size else 0.0
    im = ax.imshow(s, cmap=SEQ, vmin=vmin, vmax=1.0)
    names = [cmp.labels[i] for i in order]
    ax.set_xticks(range(cmp.n), names, rotation=40, ha="right", fontsize=10.5)
    ax.set_yticks(range(cmp.n), names, fontsize=10.5)
    for labels in (ax.get_xticklabels(), ax.get_yticklabels()):
        labels[0].set_fontweight("bold")  # a consulta
        labels[0].set_color(INK)
    mid = (vmin + 1.0) / 2
    if cmp.n <= 12:
        for i in range(cmp.n):
            for j in range(cmp.n):
                ax.text(j, i, f"{s[i, j]:.2f}", ha="center", va="center", fontsize=9.5,
                        color="white" if s[i, j] > mid else INK)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(length=0)
    bar = (fig or ax.figure).colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    bar.set_label(f"score composto (escala de {vmin:.2f} a 1)", color=INK2)
    bar.outline.set_visible(False)
    ax.set_title("Todos contra todos (negrito: a consulta)", loc="left", fontsize=15, fontweight="bold")


def draw_map(ax, cmp: VectorComparison) -> None:
    ax.set_title("Mapa dos documentos (PCA 2D)", loc="left", fontsize=15, fontweight="bold")
    pts, row, xy, explained = _project(cmp, 2)
    if len(pts) < 3:
        ax.text(0.5, 0.5, "são necessários ao menos 3 documentos", ha="center", va="center", color=MUTED, transform=ax.transAxes)
        ax.set_axis_off()
        return
    q = cmp.vectors[cmp.query_pos]
    others = [i for i in pts if i != cmp.query_pos]
    sim = {i: composite(cmp, q, cmp.vectors[i]) for i in others}
    lo = max(0.0, math.floor(min(sim.values()) * 20) / 20)
    rows = [row[i] for i in others]
    ax.scatter(xy[rows, 0], xy[rows, 1], c=[sim[i] for i in others], cmap=SEQ, vmin=lo, vmax=1.0, s=190,
               edgecolors=SURFACE, linewidths=2, zorder=3)
    ax.scatter(*xy[row[cmp.query_pos]], marker="*", s=520, color=INK, edgecolors=SURFACE, linewidths=2, zorder=4)

    # pontos que quase coincidem (a mesma família) ganham UM rótulo com todos os nomes, em vez de se sobrepor
    span = float(np.ptp(xy, axis=0).max()) or 1.0
    groups: list[list[int]] = []
    for i in others:
        for group in groups:
            if np.linalg.norm(xy[row[i]] - xy[row[group[0]]]) < 0.07 * span:
                group.append(i)
                break
        else:
            groups.append([i])
    offsets = [(16, 14), (-16, -14), (18, -14), (-18, 14), (0, 26), (0, -28)]
    singles = [(xy[row[cmp.query_pos]], _query_note(cmp), True)] + [
        (xy[row[g[0]]], cmp.labels[g[0]], False) for g in groups if len(g) == 1
    ]
    for k, (point, text, is_query) in enumerate(singles):
        dx, dy = offsets[k % len(offsets)]
        ax.annotate(text, point, xytext=(dx, dy), textcoords="offset points", fontsize=10.5,
                    color=INK if is_query else INK2, fontweight="bold" if is_query else "normal",
                    ha="left" if dx >= 0 else "right", va="center", linespacing=1.3,
                    arrowprops={"arrowstyle": "-", "color": BASELINE, "linewidth": 0.8, "shrinkA": 0, "shrinkB": 8})
    # aglomerado (pontos quase coincidentes): cada documento ganha o SEU nome, numa coluna à esquerda,
    # com uma linha até o seu próprio ponto; assim os nomes não se sobrepõem e ninguém parece idêntico a ninguém
    step = 0.07 * span
    for g in (g for g in groups if len(g) > 1):
        centre = xy[[row[i] for i in g]].mean(axis=0)
        for r, i in enumerate(sorted(g, key=lambda j: -xy[row[j], 1])):  # de cima para baixo
            lx, ly = centre[0] - 0.12 * span, centre[1] + ((len(g) - 1) / 2 - r) * step
            ax.plot([xy[row[i], 0], lx], [xy[row[i], 1], ly], color=BASELINE, linewidth=0.8, zorder=2)
            ax.text(lx, ly, cmp.labels[i] + " ", ha="right", va="center", fontsize=10.5, color=INK2)
    pad = 0.18
    ax.margins(x=pad, y=pad)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(f"PC1 ({explained[0]:.0%} da variação)    PC2 ({explained[1]:.0%})", color=INK2)
    for side in ax.spines.values():
        side.set_color(GRID)
    ax.legend(
        handles=[Line2D([], [], marker="*", color="none", markerfacecolor=INK, markersize=16, label="consulta"),
                 Line2D([], [], marker="o", color="none", markerfacecolor="#3987e5", markeredgecolor=SURFACE,
                        markersize=11, label="documento (mais escuro = mais parecido)")],
        loc="upper right", frameon=False, fontsize=10.5,
    )


def draw_map3d(ax, cmp: VectorComparison) -> None:
    """Mesmo mapa do PCA 2D, com a 3ª componente: documentos que se sobrepõem em 2D podem se separar em 3D."""
    ax.set_title("Mapa 3D dos documentos (PCA)", loc="left", fontsize=15, fontweight="bold")
    pts, row, xyz, explained = _project(cmp, 3)
    if len(pts) < 4:
        ax.text2D(0.5, 0.5, "são necessários ao menos 4 documentos para o mapa 3D", ha="center", va="center",
                  color=MUTED, transform=ax.transAxes)
        ax.set_axis_off()
        return
    q = cmp.vectors[cmp.query_pos]
    others = [i for i in pts if i != cmp.query_pos]
    sim = {i: composite(cmp, q, cmp.vectors[i]) for i in others}
    lo = max(0.0, math.floor(min(sim.values()) * 20) / 20)
    rows = [row[i] for i in others]

    span = float(np.ptp(xyz, axis=0).max()) or 1.0
    floor = float(xyz[:, 2].min()) - 0.25 * span
    for point in xyz:  # linhas até o "chão": ajudam a ler a profundidade numa imagem estática
        ax.plot([point[0]] * 2, [point[1]] * 2, [floor, point[2]], color=BASELINE, linewidth=0.9, zorder=1)
    ax.scatter(*xyz[rows].T, c=[sim[i] for i in others], cmap=SEQ, vmin=lo, vmax=1.0, s=170, edgecolors=SURFACE,
               linewidths=1.5, depthshade=False, zorder=3)
    ax.scatter(*xyz[row[cmp.query_pos]], marker="*", s=480, color=INK, edgecolors=SURFACE, linewidths=1.5,
               depthshade=False, zorder=4)

    groups: list[list[int]] = []  # pontos praticamente coincidentes dividem um único rótulo
    for i in others:
        for group in groups:
            if np.linalg.norm(xyz[row[i]] - xyz[row[group[0]]]) < 0.07 * span:
                group.append(i)
                break
        else:
            groups.append([i])
    singles = [(xyz[row[cmp.query_pos]], _query_note(cmp), True)] + [
        (xyz[row[g[0]]], cmp.labels[g[0]], False) for g in groups if len(g) == 1
    ]
    for (px, py, pz), text, is_query in singles:
        # o rótulo da consulta (mais longo) cresce para a direita e sobe, para não cobrir a coluna de nomes do aglomerado
        ax.text(px, py, pz + (0.16 if is_query else 0.05) * span, text, fontsize=10.5, ha="left" if is_query else "center",
                va="bottom", linespacing=1.3, color=INK if is_query else INK2, fontweight="bold" if is_query else "normal")
    # aglomerado: cada documento ganha o SEU nome, numa coluna à esquerda, com uma linha até o seu próprio ponto
    step = 0.075 * span
    for g in (g for g in groups if len(g) > 1):
        centre = xyz[[row[i] for i in g]].mean(axis=0)
        for r, i in enumerate(sorted(g, key=lambda j: -xyz[row[j], 2])):  # de cima para baixo
            lx, ly, lz = centre[0] - 0.16 * span, centre[1], centre[2] + ((len(g) - 1) / 2 - r) * step
            px, py, pz = xyz[row[i]]
            ax.plot([px, lx], [py, ly], [pz, lz], color=BASELINE, linewidth=0.8, zorder=2)
            ax.text(lx, ly, lz, cmp.labels[i] + " ", ha="right", va="center", fontsize=10.5, color=INK2)

    for axis, name, share in zip((ax.xaxis, ax.yaxis, ax.zaxis), ("PC1", "PC2", "PC3"), explained):
        axis.set_pane_color(matplotlib.colors.to_rgba(SURFACE))
        axis.pane.set_edgecolor(GRID)
        axis.set_ticklabels([])  # as unidades da projeção não significam nada; o que vale é a distância relativa
        axis.set_label_text(f"{name} ({share:.0%})")
    ax.set_zlim(floor, float(xyz[:, 2].max()) + 0.25 * span)
    ax.set_box_aspect((1, 1, 0.8))
    ax.view_init(elev=22, azim=-58)
    ax.legend(
        handles=[Line2D([], [], marker="*", color="none", markerfacecolor=INK, markersize=16, label="consulta"),
                 Line2D([], [], marker="o", color="none", markerfacecolor="#3987e5", markeredgecolor=SURFACE,
                        markersize=11, label="documento (mais escuro = mais parecido)")],
        loc="upper right", frameon=False, fontsize=10.5,
    )


def cluster_members(cmp: VectorComparison) -> list[int]:
    """O maior aglomerado de documentos que se sobrepõem no mapa 2D (os que dividem a coluna de nomes). Vazio se não há."""
    pts, row, xy, _ = _project(cmp, 2)
    span = float(np.ptp(xy, axis=0).max()) or 1.0
    groups: list[list[int]] = []
    for i in (p for p in pts if p != cmp.query_pos):
        for group in groups:
            if np.linalg.norm(xy[row[i]] - xy[row[group[0]]]) < 0.07 * span:
                group.append(i)
                break
        else:
            groups.append([i])
    biggest = max(groups, key=len, default=[])
    return biggest if len(biggest) >= 3 else []


def _zoom_labels(cmp: VectorComparison, members: list[int], sim: dict[int, float]) -> dict[int, str]:
    return {i: f"{cmp.labels[i]}\n{sim[i]:.4f}" for i in members}


def draw_zoom2d(fig, ax, cmp: VectorComparison, members: list[int]) -> None:
    """PCA 2D recalculado SÓ com o aglomerado: a escala passa a ser a das diferenças entre eles, e os pontos se separam."""
    ax.set_title("Zoom no grupo quase idêntico (PCA 2D)", loc="left", fontsize=15, fontweight="bold")
    pts, row, xy, explained = _project(cmp, 2, members)
    q = cmp.vectors[cmp.query_pos]
    sim = {i: composite(cmp, q, cmp.vectors[i]) for i in members}
    labels = _zoom_labels(cmp, members, sim)
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=[sim[i] for i in pts], cmap=SEQ, vmin=min(sim.values()) - 0.002,
                    vmax=max(sim.values()) + 0.002, s=300, edgecolors=SURFACE, linewidths=2, zorder=3)
    offsets = [(18, 16), (-18, -18), (20, -18), (-20, 18), (0, 30), (0, -34)]
    for k, i in enumerate(sorted(pts, key=lambda j: (xy[row[j], 0], xy[row[j], 1]))):
        dx, dy = offsets[k % len(offsets)]
        ax.annotate(labels[i], xy[row[i]], xytext=(dx, dy), textcoords="offset points", fontsize=10.5, color=INK2,
                    ha="left" if dx >= 0 else "right", va="center", linespacing=1.25,
                    arrowprops={"arrowstyle": "-", "color": BASELINE, "linewidth": 0.8, "shrinkA": 0, "shrinkB": 9})
    ax.margins(x=0.35, y=0.35)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(f"PC1 ({explained[0]:.0%})    PC2 ({explained[1]:.0%})    PCA recalculado só com estes {len(members)} documentos",
                  color=INK2)
    for side in ax.spines.values():
        side.set_color(GRID)
    bar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    bar.set_label("score composto com a consulta (escala só deste grupo)", color=INK2)
    bar.outline.set_visible(False)


def draw_zoom3d(ax, cmp: VectorComparison, members: list[int]) -> None:
    ax.set_title("Zoom no grupo (PCA 3D)", loc="left", fontsize=15, fontweight="bold")
    pts, row, xyz, explained = _project(cmp, 3, members)
    if len(pts) < 4:
        ax.text2D(0.5, 0.5, "são necessários ao menos 4 documentos no grupo para o 3D", ha="center", va="center",
                  color=MUTED, transform=ax.transAxes)
        ax.set_axis_off()
        return
    q = cmp.vectors[cmp.query_pos]
    sim = {i: composite(cmp, q, cmp.vectors[i]) for i in members}
    labels = _zoom_labels(cmp, members, sim)
    span = float(np.ptp(xyz, axis=0).max()) or 1.0
    floor = float(xyz[:, 2].min()) - 0.25 * span
    for point in xyz:
        ax.plot([point[0]] * 2, [point[1]] * 2, [floor, point[2]], color=BASELINE, linewidth=0.9, zorder=1)
    ax.scatter(*xyz.T, c=[sim[i] for i in pts], cmap=SEQ, vmin=min(sim.values()) - 0.002, vmax=max(sim.values()) + 0.002,
               s=240, edgecolors=SURFACE, linewidths=1.5, depthshade=False, zorder=3)
    for k, i in enumerate(sorted(pts, key=lambda j: xyz[row[j], 0])):
        px, py, pz = xyz[row[i]]
        dz = (0.14 if k % 2 == 0 else 0.38) * span  # vizinhos alternam a altura do rótulo para não se tocarem
        ax.plot([px, px], [py, py], [pz, pz + dz], color=BASELINE, linewidth=0.8, zorder=2)
        ax.text(px, py, pz + dz, labels[i], fontsize=10, ha="center", va="bottom", linespacing=1.25, color=INK2)
    for axis, name, share in zip((ax.xaxis, ax.yaxis, ax.zaxis), ("PC1", "PC2", "PC3"), explained):
        axis.set_pane_color(matplotlib.colors.to_rgba(SURFACE))
        axis.pane.set_edgecolor(GRID)
        axis.set_ticklabels([])
        axis.set_label_text(f"{name} ({share:.0%})")
    ax.set_zlim(floor, float(xyz[:, 2].max()) + 0.7 * span)
    ax.set_box_aspect((1, 1, 0.8))
    ax.view_init(elev=22, azim=-58)


# ---- figuras -----------------------------------------------------------------------------------------------

def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return buf.getvalue()


def render_vector_panels(cmp: VectorComparison) -> dict[str, bytes]:
    """{"combined", "bars", "matrix", "map", "map3d"} -> PNG, prontos para colar nos slides."""
    out: dict[str, bytes] = {}
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(12, 1.6 + 0.85 * min(cmp.n, 10)))
        draw_bars(ax, cmp)
        out["bars"] = _png(fig)

        fig, ax = plt.subplots(figsize=(8.5 + 0.2 * cmp.n, 7 + 0.15 * cmp.n))
        draw_matrix(ax, cmp, fig)
        out["matrix"] = _png(fig)

        fig, ax = plt.subplots(figsize=(10, 7.5))
        draw_map(ax, cmp)
        out["map"] = _png(fig)

        fig = plt.figure(figsize=(10, 8))
        draw_map3d(fig.add_subplot(projection="3d"), cmp)
        out["map3d"] = _png(fig)

        members = cluster_members(cmp)  # só existe se há um aglomerado de ao menos 3 documentos sobrepostos
        if members:
            fig = plt.figure(figsize=(19, 8))
            gs = fig.add_gridspec(1, 2, wspace=0.18)
            draw_zoom2d(fig, fig.add_subplot(gs[0, 0]), cmp, members)
            draw_zoom3d(fig.add_subplot(gs[0, 1], projection="3d"), cmp, members)
            out["zoom"] = _png(fig)

        rows = 3 if members else 2  # com aglomerado, o zoom entra como 3ª linha (os mapas gerais colapsam o grupo num ponto)
        fig = plt.figure(figsize=(21, 13 + (7 if members else 0)))
        gs = fig.add_gridspec(rows, 2, hspace=0.42, wspace=0.34)
        draw_bars(fig.add_subplot(gs[0, 0]), cmp)
        draw_matrix(fig.add_subplot(gs[0, 1]), cmp, fig)
        draw_map(fig.add_subplot(gs[1, 0]), cmp)
        draw_map3d(fig.add_subplot(gs[1, 1], projection="3d"), cmp)
        if members:
            draw_zoom2d(fig, fig.add_subplot(gs[2, 0]), cmp, members)
            draw_zoom3d(fig.add_subplot(gs[2, 1], projection="3d"), cmp, members)
        fig.suptitle(f"Comparação de vetores — consulta: {cmp.query_label}", x=0.01, ha="left", fontsize=19, fontweight="bold")
        out["combined"] = _png(fig)
    return out
