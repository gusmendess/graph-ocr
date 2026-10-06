import io
import textwrap

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

from app.api.domain.alignment import Alignment  # noqa: E402
from app.api.domain.bbox import BBox  # noqa: E402
from app.api.domain.enums import WordStatus  # noqa: E402
from app.api.models.schemas import WordMatch  # noqa: E402
from app.api.use_cases.analyze_document import AnalyzeOutcome  # noqa: E402
from app.api.use_cases.compare_documents import CompareOutcome  # noqa: E402

COLORS = {
    WordStatus.MATCHED: "#2ca02c",
    WordStatus.MOVED: "#ff7f0e",
    WordStatus.MODIFIED: "#d62728",
    WordStatus.REMOVED: "#9467bd",
    WordStatus.ADDED: "#1f77b4",
    WordStatus.CROPPED: "#7f7f7f",
}
_LABELS = {
    WordStatus.MATCHED: "igual (mesmo texto e lugar)",
    WordStatus.MOVED: "deslocada",
    WordStatus.MODIFIED: "alterada (mesmo lugar, outro texto)",
    WordStatus.REMOVED: "removida (só em A)",
    WordStatus.ADDED: "adicionada (só em B)",
    WordStatus.CROPPED: "fora do recorte de B",
}


def _png(fig, dpi: int = 90) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _align_b(outcome: CompareOutcome) -> np.ndarray:
    """B sobreposta a A (warp pela transformação estimada); sem alinhamento, só ajusta o tamanho."""
    img_a, img_b, al = outcome.image_a, outcome.image_b, outcome.alignment
    h, w = img_a.shape[:2]
    if al is not None and al.converged:
        return cv2.warpAffine(img_b, al.matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    return img_b if img_b.shape == img_a.shape else cv2.resize(img_b, (w, h))


def _pixel_diff(outcome: CompareOutcome) -> np.ndarray:
    diff = cv2.absdiff(cv2.cvtColor(outcome.image_a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(_align_b(outcome), cv2.COLOR_BGR2GRAY))
    return cv2.dilate(diff, np.ones((7, 7), np.uint8))  # dilata: diferenças finas somem na miniatura


def _ocr_diff(outcome: CompareOutcome) -> np.ndarray:
    """Mapa de calor geométrico: só as palavras que mudaram acendem (intensidade = suspeita)."""
    h, w = outcome.image_a.shape[:2]
    al = outcome.alignment or Alignment.identity()
    canvas = np.zeros((h, w), np.uint8)
    for m in outcome.result.matches:
        if m.status in (WordStatus.MATCHED, WordStatus.CROPPED):
            continue
        value = int(255 * max(m.suspicion, 0.3))
        boxes = [m.a.bbox if m.a else None, al.apply_bbox(m.b.bbox) if m.b else None]
        for b in boxes:
            if b is not None:
                canvas[max(int(b.y0), 0) : int(b.y1), max(int(b.x0), 0) : int(b.x1)] = value
    return cv2.dilate(canvas, np.ones((5, 5), np.uint8))


def _draw_words(ax, image: np.ndarray, matches: list[WordMatch], side: str) -> None:
    ax.imshow(cv2.addWeighted(_rgb(image), 0.45, np.full_like(_rgb(image), 255), 0.55, 0))
    for m in matches:
        word = m.a if side == "a" else m.b
        if word is None:
            continue
        b, color = word.bbox, COLORS[m.status]
        loud = m.status != WordStatus.MATCHED
        ax.add_patch(
            Rectangle(
                (b.x0, b.y0), b.width, b.height,
                fill=loud, facecolor=color, alpha=0.35 if loud else 1.0,
                edgecolor=color, linewidth=2.2 if loud else 0.6,
            )
        )


def _heatmap(ax, fig, data: np.ndarray, title: str) -> None:
    im = ax.imshow(data, cmap="hot", vmin=0, vmax=255)
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)


def render_comparison(outcome: CompareOutcome, label_a: str = "A", label_b: str = "B") -> bytes:
    """2x3: [original] A | B | |A-B|   e   [OCR + bboxes] A | B | |A-B| geométrico."""
    result = outcome.result
    fig, axes = plt.subplots(2, 3, figsize=(15, 33), gridspec_kw={"hspace": 0.04, "wspace": 0.12})

    axes[0, 0].imshow(_rgb(outcome.image_a))
    axes[0, 0].set_title(f"[original] A: {label_a}", fontsize=10)
    axes[0, 1].imshow(_rgb(outcome.image_b))
    axes[0, 1].set_title(f"[original] B: {label_b}", fontsize=10)
    aligned = outcome.alignment is not None and outcome.alignment.converged
    _heatmap(axes[0, 2], fig, _pixel_diff(outcome), "|A-B| original (pixels, B alinhada a A)" if aligned else "|A-B| original (pixels)")

    if outcome.words_a is not None:
        engine = result.ocr_engine
        _draw_words(axes[1, 0], outcome.image_a, result.matches, "a")
        axes[1, 0].set_title(f"[{engine}_words] A pré-processado", fontsize=10)
        _draw_words(axes[1, 1], outcome.image_b, result.matches, "b")
        axes[1, 1].set_title(f"[{engine}_words] B pré-processado", fontsize=10)
        _heatmap(axes[1, 2], fig, _ocr_diff(outcome), f"|A-B| {engine}_words (palavras que mudaram)")
    else:
        for ax in axes[1]:
            ax.text(0.5, 0.5, f"OCR não executado\n(decidido por {result.decided_by})", ha="center", va="center")

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.legend(
        handles=[Patch(color=COLORS[s], label=_LABELS[s]) for s in WordStatus],
        loc="lower center", ncol=6, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.085),
    )
    head = f"Veredito: {result.verdict.value.upper()}  |  score {result.score:.3f}  |  decidido por {result.decided_by}"
    fig.suptitle(head + "\n" + "\n".join(textwrap.wrap(result.summary, 140)), fontsize=12, y=0.915)
    return _png(fig)


def _border_point(box: BBox, toward: tuple[float, float]) -> tuple[float, float]:
    """Ponto da borda do retângulo na direção de `toward` (a aresta nasce na borda, não no centro)."""
    cx, cy = box.center
    dx, dy = toward[0] - cx, toward[1] - cy
    if dx == 0 and dy == 0:
        return cx, cy
    t = min((box.width / 2) / abs(dx) if dx else np.inf, (box.height / 2) / abs(dy) if dy else np.inf)
    return cx + dx * min(t, 1.0), cy + dy * min(t, 1.0)


def render_graph_overlay(outcome: CompareOutcome, label_a: str = "A", label_b: str = "B") -> bytes | None:
    """Grafos de palavras sobrepostos no referencial de A: nós = retângulos (bboxes), arestas = vizinhança k-NN.

    A em azul (traço cheio), B alinhado em laranja (tracejado); palavras alteradas em vermelho.
    """
    if outcome.graph_a is None or outcome.graph_b is None:
        return None
    al = outcome.alignment or Alignment.identity()
    h, w = outcome.image_a.shape[:2]
    fig, ax = plt.subplots(figsize=(10, 10 * h / w))
    ax.imshow(np.full((h, w, 3), 255, np.uint8))

    boxes_a = [outcome.graph_a.nodes[n]["word"].bbox for n in sorted(outcome.graph_a.nodes)]
    boxes_b = [al.apply_bbox(outcome.graph_b.nodes[n]["word"].bbox) for n in sorted(outcome.graph_b.nodes)]
    texts_a = [outcome.graph_a.nodes[n]["word"].text for n in sorted(outcome.graph_a.nodes)]

    edits = (WordStatus.MODIFIED, WordStatus.ADDED, WordStatus.REMOVED, WordStatus.MOVED)
    changed_a = {m.a_index for m in outcome.result.matches if m.status in edits and m.a_index is not None}
    changed_b = {m.b_index for m in outcome.result.matches if m.status in edits and m.b_index is not None}

    def draw(graph, boxes, color, style, changed, texts=None):
        for i, j in graph.edges:
            p0 = _border_point(boxes[i], boxes[j].center)
            p1 = _border_point(boxes[j], boxes[i].center)
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=1.1, alpha=0.75, zorder=2)
        for i, b in enumerate(boxes):
            hot = i in changed
            ax.add_patch(
                Rectangle(
                    (b.x0, b.y0), b.width, b.height,
                    facecolor="#d62728" if hot else color, alpha=0.35 if hot else 0.12, edgecolor="none", zorder=3,
                )
            )
            ax.add_patch(
                Rectangle(
                    (b.x0, b.y0), b.width, b.height, fill=False, linestyle=style,
                    edgecolor="#d62728" if hot else color, linewidth=2.2 if hot else 1.3, zorder=4,
                )
            )
            if texts:
                ax.text(*b.center, texts[i], ha="center", va="center", fontsize=5.5, color="#333", zorder=5, clip_on=True)

    draw(outcome.graph_a, boxes_a, "#1f77b4", "-", changed_a, texts_a)
    draw(outcome.graph_b, boxes_b, "#ff7f0e", (0, (4, 2)), changed_b)

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(
        handles=[
            Patch(facecolor="#1f77b4", alpha=0.5, label=f"grafo A: {label_a}"),
            Patch(facecolor="#ff7f0e", alpha=0.5, label=f"grafo B alinhado: {label_b}"),
            Patch(facecolor="#d62728", alpha=0.5, label="palavra alterada"),
        ],
        loc="upper right", fontsize=9,
    )
    ax.set_title("Grafos sobrepostos: nós = bboxes, arestas = vizinhança k-NN", fontsize=11)
    return _png(fig, dpi=110)


def _crop(image: np.ndarray, box: BBox | None, pad: float) -> tuple[np.ndarray, int] | None:
    """Faixa horizontal da imagem em volta da caixa; None se a caixa cai fora da imagem."""
    if box is None:
        return None
    h = image.shape[0]
    y0, y1 = int(max(box.y0 - pad, 0)), int(min(box.y1 + pad, h))
    if y1 - y0 < 4 or not 0 <= box.center[1] < h:
        return None
    return image[y0:y1], y0


def _side_box(m: WordMatch, side: str, alignment: Alignment | None) -> BBox | None:
    """Caixa do campo NA imagem do lado pedido (cada imagem tem o seu enquadramento).

    Se a palavra existe nesse lado, usa a dela. Se não (só em A / só em B), projeta a do outro lado
    pelo alinhamento, desde que ele tenha convergido; senão não há onde recortar.
    """
    own, other = (m.a, m.b) if side == "a" else (m.b, m.a)
    if own is not None:
        return own.bbox
    if other is None or alignment is None or not alignment.converged:
        return None
    if side == "a":  # palavra só em B -> posição em A
        return alignment.apply_bbox(other.bbox)
    inverse = cv2.invertAffineTransform(alignment.matrix.astype(np.float32)).astype(float)
    return Alignment(inverse).apply_bbox(other.bbox)  # palavra só em A -> posição em B


def render_evidence_zoom(outcome: CompareOutcome, label_a: str = "A", label_b: str = "B", max_items: int = 6) -> bytes | None:
    """Recorte ampliado de cada campo alterado, A ao lado de B, com overlay vermelho só no campo."""
    edits = (WordStatus.MODIFIED, WordStatus.ADDED, WordStatus.REMOVED)  # recorte e deslocamento não são edição
    changed = sorted((m for m in outcome.result.matches if m.status in edits), key=lambda m: m.suspicion, reverse=True)
    changed = changed[:max_items]
    if not changed:
        return None

    fig, axes = plt.subplots(len(changed), 2, figsize=(12, 2.4 * len(changed)), squeeze=False)
    for row, m in zip(axes, changed):
        ref_h = (m.b or m.a).bbox.height
        for ax, image, side, word, label in (
            (row[0], outcome.image_a, "a", m.a, label_a),
            (row[1], outcome.image_b, "b", m.b, label_b),
        ):
            crop = _crop(image, _side_box(m, side, outcome.alignment), pad=2.0 * ref_h)
            ax.set_xticks([])
            ax.set_yticks([])
            if crop is None:
                ax.text(0.5, 0.5, "(sem posição correspondente nesta imagem)", ha="center", va="center", fontsize=9, color="#666")
                ax.set_title(f"{label}: '{word.text}'" if word else f"{label}: (ausente)", fontsize=10)
                continue
            strip, y_off = crop
            ax.imshow(_rgb(strip))
            if word is not None:
                b = word.bbox
                ax.add_patch(
                    Rectangle((b.x0, b.y0 - y_off), b.width, b.height, fill=True, facecolor="red", alpha=0.25,
                              edgecolor="red", linewidth=2.5)
                )
            ax.set_title(f"{label}: '{word.text}'" if word else f"{label}: (ausente)", fontsize=10)
        d = m.deltas
        extra = f" | Δaltura {d.height_pct:+.1f}%  Δlargura/char {d.char_width_pct:+.1f}%  Δbaseline {d.baseline_px:+.1f}px  Δconf {d.conf:+.3f}" if d else ""
        row[0].set_ylabel(f"{m.status.value}\n{m.kind.value}\nsuspeita {m.suspicion:.2f}", fontsize=9)
        row[1].text(0.0, -0.12, extra.strip(" |"), transform=row[1].transAxes, fontsize=8, color="#444")
    fig.tight_layout()
    return _png(fig)


def render_document_report(outcome: AnalyzeOutcome, label: str = "documento") -> bytes:
    """Documento isolado: palavras que violam regras em vermelho; sinal geométrico fraco em laranja."""
    report = outcome.report
    flagged = {i for f in report.flags for i in f.word_indexes}
    weak = {a.index for a in report.anomalies if a.index is not None} - flagged
    h, w = outcome.image.shape[:2]

    fig, ax = plt.subplots(figsize=(8, 8 * h / w))
    ax.imshow(cv2.addWeighted(_rgb(outcome.image), 0.6, np.full((h, w, 3), 255, np.uint8), 0.4, 0))
    for i, word in enumerate(outcome.words):
        b = word.bbox
        if i in flagged:
            ax.add_patch(Rectangle((b.x0, b.y0), b.width, b.height, facecolor="red", alpha=0.3, edgecolor="red", linewidth=2.5))
        elif i in weak:
            ax.add_patch(Rectangle((b.x0, b.y0), b.width, b.height, fill=False, edgecolor="#ff7f0e", linewidth=1.6, linestyle="--"))
        else:
            ax.add_patch(Rectangle((b.x0, b.y0), b.width, b.height, fill=False, edgecolor="#2ca02c", linewidth=0.5, alpha=0.6))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(
        handles=[
            Patch(facecolor="red", alpha=0.4, label="viola regra de consistência"),
            Patch(facecolor="none", edgecolor="#ff7f0e", linestyle="--", label="anomalia geométrica (sinal fraco)"),
            Patch(facecolor="none", edgecolor="#2ca02c", label="sem achados"),
        ],
        loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=8, frameon=False,
    )
    head = f"{label}  |  veredito: {report.verdict.value.upper()}  |  {report.n_words} palavras ({report.ocr_engine})"
    fig.suptitle(head + "\n" + "\n".join(textwrap.wrap(report.summary, 100)), fontsize=10, y=0.94)
    return _png(fig, dpi=100)
