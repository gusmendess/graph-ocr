from collections import defaultdict
from dataclasses import dataclass

import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment

from app.api.domain.bbox import BBox
from app.api.domain.entities import Word
from app.api.domain.enums import WordStatus
from app.api.models.schemas import WordMatch
from app.api.services.embedding_service import Embedder, cosine_matrix
from app.api.services.graph_service import build_word_graph


@dataclass
class Line:
    indices: list[int]  # índices das palavras (na lista do OCR) que formam a linha
    word: Word  # linha como "palavra" única: texto, bbox união, confiança média


def group_lines(words: list[Word]) -> list[Line]:
    """Agrupa palavras por linha do OCR (campo). Sem line_id, cada palavra é uma linha."""
    groups: dict[object, list[int]] = defaultdict(list)
    for i, w in enumerate(words):
        groups[w.line_id if w.line_id is not None else ("solo", i)].append(i)
    lines = []
    for idx in groups.values():
        idx.sort(key=lambda i: words[i].bbox.x0)
        ws = [words[i] for i in idx]
        box = BBox(min(w.bbox.x0 for w in ws), min(w.bbox.y0 for w in ws), max(w.bbox.x1 for w in ws), max(w.bbox.y1 for w in ws))
        lines.append(Line(idx, Word(" ".join(w.text for w in ws), box, float(np.mean([w.conf for w in ws])))))
    return sorted(lines, key=lambda ln: (ln.word.bbox.y0, ln.word.bbox.x0))


def _neighbor_mean(vecs: np.ndarray, graph: nx.Graph) -> np.ndarray:
    """Vetor médio dos vizinhos de cada nó (o próprio nó fica de fora: é o contexto dele)."""
    ctx = np.zeros_like(vecs)
    for node in graph.nodes:
        neighbors = list(graph.neighbors(node))
        if neighbors:
            ctx[node] = vecs[neighbors].mean(axis=0)
    return ctx


def _contextual(vecs: np.ndarray, ctx: np.ndarray, weight: float) -> np.ndarray:
    """[vetor próprio ; peso * contexto], L2-normalizado (peso 0 = só o texto)."""
    both = np.hstack([vecs, weight * ctx])
    return both / np.maximum(np.linalg.norm(both, axis=1, keepdims=True), 1e-12)


def match_by_vectors(
    words_a: list[Word],
    words_b: list[Word],
    embedder: Embedder,
    knn_k: int = 4,
    cos_same: float = 0.995,
    cos_edit: float = 0.3,
    context_weight: float = 0.3,
    slot_overlap: float = 0.5,
) -> list[WordMatch]:
    """Pareia A e B só por similaridade de cosseno entre vetores de texto (nada de igualdade exata).

    1. Cada linha do OCR vira um nó com vetor de texto; o grafo k-NN das linhas dá o contexto
       (média dos vizinhos). O pareamento de linhas é a atribuição ótima sobre o cosseno contextual.
    2. Dentro de cada par de linhas, o mesmo vale para as palavras (cosseno + atribuição).
       cosseno >= cos_same -> MATCHED;  cos_edit <= cosseno < cos_same -> MODIFIED (quase igual);
       sobras no mesmo par de linhas ocupam o mesmo "slot" -> MODIFIED.
    3. Slot pelo grafo (propagação a partir de âncoras): linhas com cosseno ~1 são âncoras. Uma linha
       cujo texto mudou por completo (cosseno < cos_edit) ocupa o mesmo campo se os vizinhos k-NN que
       são âncoras coincidem (Jaccard >= slot_overlap). Vizinhos também editados são ignorados, então
       edições adjacentes não se contaminam. É o caso do fraudador que troca o valor inteiro
       (ex.: código da transação, ID Pix) -> MODIFIED.
    4. O que continuar sem par -> REMOVED (só em A) / ADDED (só em B).

    Não usa geometria absoluta, então é indiferente a deslocamento, escala, rotação ou perspectiva.
    """
    lines_a, lines_b = group_lines(words_a), group_lines(words_b)
    if not lines_a or not lines_b:
        return _unmatched(words_a, words_b, range(len(words_a)), range(len(words_b)))

    va = embedder.embed([ln.word.text for ln in lines_a])
    vb = embedder.embed([ln.word.text for ln in lines_b])
    self_cos = cosine_matrix(va, vb)
    graph_a = build_word_graph([ln.word for ln in lines_a], knn_k)
    graph_b = build_word_graph([ln.word for ln in lines_b], knn_k)
    ha = _contextual(va, _neighbor_mean(va, graph_a), context_weight)
    hb = _contextual(vb, _neighbor_mean(vb, graph_b), context_weight)
    rows, cols = linear_sum_assignment(-(ha @ hb.T))

    pairs = [(r, c) for r, c in zip(rows, cols) if self_cos[r, c] >= cos_edit]  # iguais ou quase iguais
    anchors = {r: c for r, c in pairs if self_cos[r, c] >= cos_same}
    pairs += _slot_pairs(graph_a, graph_b, anchors, {r for r, _ in pairs}, {c for _, c in pairs},
                         self_cos, slot_overlap)

    matches: list[WordMatch] = []
    used_a: set[int] = set()
    used_b: set[int] = set()
    for r, c in pairs:
        matches += _match_words_in_line(words_a, words_b, lines_a[r], lines_b[c], embedder, cos_same, cos_edit)
        used_a.update(lines_a[r].indices)
        used_b.update(lines_b[c].indices)

    matches += _unmatched(words_a, words_b, [i for i in range(len(words_a)) if i not in used_a],
                          [j for j in range(len(words_b)) if j not in used_b])
    matches.sort(key=_reading_order)
    return matches


def _slot_pairs(graph_a, graph_b, anchors, taken_a, taken_b, self_cos, slot_overlap) -> list[tuple[int, int]]:
    """Linhas ainda sem par que ocupam o mesmo campo: mesmos vizinhos-âncora nos dois grafos."""
    free_a = [i for i in graph_a.nodes if i not in taken_a]
    free_b = [j for j in graph_b.nodes if j not in taken_b]
    if not free_a or not free_b or not anchors:
        return []
    anchored_b = set(anchors.values())
    jac = np.zeros((len(free_a), len(free_b)))
    for x, i in enumerate(free_a):
        side_a = {anchors[n] for n in graph_a.neighbors(i) if n in anchors}
        for y, j in enumerate(free_b):
            side_b = {n for n in graph_b.neighbors(j) if n in anchored_b}
            union = side_a | side_b
            jac[x, y] = len(side_a & side_b) / len(union) if union else 0.0
    # desempate por semelhança de texto (só importa quando dois campos livres têm o mesmo contexto)
    rows, cols = linear_sum_assignment(-(jac + 0.01 * self_cos[np.ix_(free_a, free_b)]))
    return [(free_a[r], free_b[c]) for r, c in zip(rows, cols) if jac[r, c] >= slot_overlap]


def _match_words_in_line(words_a, words_b, line_a: Line, line_b: Line, embedder, cos_same, cos_edit) -> list[WordMatch]:
    ia, ib = line_a.indices, line_b.indices
    cos = cosine_matrix(embedder.embed([words_a[i].text for i in ia]), embedder.embed([words_b[j].text for j in ib]))
    rows, cols = linear_sum_assignment(-cos)

    out: list[WordMatch] = []
    left_a, left_b = list(range(len(ia))), list(range(len(ib)))
    for r, c in zip(rows, cols):
        if cos[r, c] < cos_edit:
            continue
        status = WordStatus.MATCHED if cos[r, c] >= cos_same else WordStatus.MODIFIED
        out.append(_pair(status, ia[r], ib[c], words_a, words_b, float(cos[r, c])))
        left_a.remove(r)
        left_b.remove(c)

    # sobras do mesmo par de linhas: ocupam o mesmo slot (texto trocado por algo sem semelhança)
    for r, c in zip(left_a, left_b):
        out.append(_pair(WordStatus.MODIFIED, ia[r], ib[c], words_a, words_b, float(cos[r, c])))
    extra_a, extra_b = left_a[len(left_b):], left_b[len(left_a):]
    out += _unmatched(words_a, words_b, [ia[r] for r in extra_a], [ib[c] for c in extra_b])
    return out


def _pair(status, i, j, words_a, words_b, cosine) -> WordMatch:
    return WordMatch(status=status, a=words_a[i], b=words_b[j], a_index=i, b_index=j, cosine=cosine)


def _unmatched(words_a, words_b, idx_a, idx_b) -> list[WordMatch]:
    out = [WordMatch(status=WordStatus.REMOVED, a=words_a[i], a_index=i) for i in idx_a]
    out += [WordMatch(status=WordStatus.ADDED, b=words_b[j], b_index=j) for j in idx_b]
    return out


def _reading_order(m: WordMatch) -> tuple[float, float]:
    word = m.b or m.a
    return (round(word.bbox.center[1] / 10), word.bbox.x0)
