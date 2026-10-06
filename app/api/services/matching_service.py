from collections import defaultdict

import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment

from app.api.domain.alignment import Alignment
from app.api.domain.entities import Word
from app.api.domain.enums import WordStatus
from app.api.models.schemas import WordMatch
from app.api.services.graph_service import jaccard, neighbor_keys
from app.api.utils import normalize_text


def match_words(
    words_a: list[Word],
    words_b: list[Word],
    graph_a: nx.Graph,
    graph_b: nx.Graph,
    iou_match: float = 0.5,
    iou_slot: float = 0.3,
    alignment: Alignment | None = None,
) -> list[WordMatch]:
    """Pareia as palavras de A e B, com B já sobreposto a A pelo alinhamento de grafos.

    1. Conteúdo: mesmo texto. Se o texto se repete, a escolha é feita por IoU da bbox (no referencial
       de A) + Jaccard dos vizinhos k-NN (contexto espacial compartilhado).
       -> MATCHED (mesmo lugar) ou MOVED (mesmo texto, lugar diferente).
    2. Posição: o que sobrou é pareado por IoU. Mesmo "slot" com texto diferente -> MODIFIED.
       O que continua sem par -> REMOVED (só em A) / ADDED (só em B).
    """
    alignment = alignment or Alignment.identity()
    box_a = [w.bbox for w in words_a]
    box_b = [alignment.apply_bbox(w.bbox) for w in words_b]
    nk_a = [neighbor_keys(graph_a, i) for i in range(len(words_a))]
    nk_b = [neighbor_keys(graph_b, j) for j in range(len(words_b))]

    matches: list[WordMatch] = []
    used_a: set[int] = set()
    used_b: set[int] = set()

    groups_a: dict[str, list[int]] = defaultdict(list)
    groups_b: dict[str, list[int]] = defaultdict(list)
    for i, w in enumerate(words_a):
        groups_a[normalize_text(w.text)].append(i)
    for j, w in enumerate(words_b):
        groups_b[normalize_text(w.text)].append(j)

    for key, idx_a in groups_a.items():
        idx_b = groups_b.get(key)
        if not idx_b:
            continue
        iou = np.array([[box_a[i].iou(box_b[j]) for j in idx_b] for i in idx_a])
        jac = np.array([[jaccard(nk_a[i], nk_b[j]) for j in idx_b] for i in idx_a])
        rows, cols = linear_sum_assignment(-(0.6 * iou + 0.4 * jac))
        unique = len(idx_a) == 1 and len(idx_b) == 1  # texto único nos dois lados: par inequívoco
        for r, c in zip(rows, cols):
            if not unique and iou[r, c] < iou_slot and jac[r, c] < 0.5:
                continue
            i, j = idx_a[r], idx_b[c]
            status = WordStatus.MATCHED if iou[r, c] >= iou_match else WordStatus.MOVED
            matches.append(_pair(status, i, j, words_a, words_b, float(iou[r, c]), float(jac[r, c])))
            used_a.add(i)
            used_b.add(j)

    left_a = [i for i in range(len(words_a)) if i not in used_a]
    left_b = [j for j in range(len(words_b)) if j not in used_b]
    if left_a and left_b:
        iou = np.array([[box_a[i].iou(box_b[j]) for j in left_b] for i in left_a])
        rows, cols = linear_sum_assignment(-iou)
        for r, c in zip(rows, cols):
            if iou[r, c] < iou_slot:
                continue
            i, j = left_a[r], left_b[c]
            jac = jaccard(nk_a[i], nk_b[j])
            matches.append(_pair(WordStatus.MODIFIED, i, j, words_a, words_b, float(iou[r, c]), jac))
            used_a.add(i)
            used_b.add(j)

    for i in range(len(words_a)):
        if i not in used_a:
            matches.append(WordMatch(status=WordStatus.REMOVED, a=words_a[i], a_index=i))
    for j in range(len(words_b)):
        if j not in used_b:
            matches.append(WordMatch(status=WordStatus.ADDED, b=words_b[j], b_index=j))

    matches.sort(key=_reading_order)
    return matches


def _pair(status, i, j, words_a, words_b, iou, jac) -> WordMatch:
    return WordMatch(
        status=status, a=words_a[i], b=words_b[j], a_index=i, b_index=j, iou=iou, neighbor_jaccard=jac
    )


def _reading_order(m: WordMatch) -> tuple[float, float]:
    word = m.b or m.a
    return (round(word.bbox.center[1] / 10), word.bbox.x0)


def similarity_score(matches: list[WordMatch]) -> float:
    """Fração do conteúdo casado (palavra + posição) entre os dois documentos."""
    # cada par (inclusive MODIFIED) ou sobra (REMOVED/ADDED) conta como uma unidade de conteúdo;
    # o que ficou fora do recorte não entra na conta: B nunca teve a chance de mostrar essas palavras
    scored = [m for m in matches if m.status != WordStatus.CROPPED]
    if not scored:
        return 0.0
    return sum(1 for m in scored if m.status == WordStatus.MATCHED) / len(scored)
