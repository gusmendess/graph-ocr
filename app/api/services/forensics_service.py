import re
import statistics

import networkx as nx

from app.api.domain.alignment import Alignment
from app.api.domain.entities import Word
from app.api.domain.enums import FieldKind, WordStatus
from app.api.models.schemas import AnomalyWord, WordDeltas, WordMatch
from app.api.services.graph_service import same_line_neighbors

_DATE = re.compile(r"(?<!\d)\d{1,2}/\d{1,2}/\d{2,4}(?!\d)")
_TIME = re.compile(r"(?<!\d)\d{1,2}:\d{2}(:\d{2})?(?!\d)")
_MONEY = re.compile(r"^(R\$|\d[\d.]*,\d{2}$)")


def classify_kind(text: str) -> FieldKind:
    if _DATE.search(text):
        return FieldKind.DATE
    if _TIME.search(text):
        return FieldKind.TIME
    if _MONEY.match(text):
        return FieldKind.MONEY
    if len(text) >= 10 and sum(c.isdigit() for c in text) >= 6:
        return FieldKind.ID
    return FieldKind.TEXT


def char_width(word: Word) -> float:
    return word.bbox.width / max(len(word.text), 1)


def pair_deltas(a: Word, b: Word, alignment: Alignment | None = None) -> WordDeltas:
    """Diferenças B - A medidas no referencial de A (senão o deslocamento global vira "sinal")."""
    box = (alignment or Alignment.identity()).apply_bbox(b.bbox)
    return WordDeltas(
        height_pct=100 * (box.height - a.bbox.height) / max(a.bbox.height, 1e-6),
        char_width_pct=100 * (box.width / max(len(b.text), 1) - char_width(a)) / max(char_width(a), 1e-6),
        baseline_px=box.y1 - a.bbox.y1,
        conf=b.conf - a.conf,
    )


def geometry_anomaly(word: Word, neighbors: list[Word]) -> float:
    """Quanto a palavra destoa dos vizinhos da mesma linha (altura, largura por caractere, baseline, confiança).

    Não precisa de template: compara a palavra com o resto do mesmo documento. Retorna 0-1.
    É um sinal fraco sozinho (fontes proporcionais variam); serve como evidência complementar.
    """
    if len(neighbors) < 2:
        return 0.0
    med_h = statistics.median(n.bbox.height for n in neighbors)
    med_cw = statistics.median(char_width(n) for n in neighbors)
    med_base = statistics.median(n.bbox.y1 for n in neighbors)
    med_conf = statistics.median(n.conf for n in neighbors)
    zs = [
        abs(word.bbox.height - med_h) / (0.1 * max(med_h, 1e-6)),
        abs(char_width(word) - med_cw) / (0.2 * max(med_cw, 1e-6)),
        abs(word.bbox.y1 - med_base) / (0.1 * max(med_h, 1e-6)),
        abs(word.conf - med_conf) / 0.02,
    ]
    return min(max((max(zs) - 1) / 4, 0.0), 1.0)


def annotate_matches(matches: list[WordMatch], graph_b: nx.Graph, alignment: Alignment | None = None) -> None:
    """Preenche kind, deltas geométricos, anomalia local e suspeita de cada diferença."""
    for m in matches:
        if m.status in (WordStatus.MATCHED, WordStatus.CROPPED):
            continue
        ref = m.b or m.a
        m.kind = classify_kind(ref.text)
        critical = m.kind != FieldKind.TEXT
        if m.a is not None and m.b is not None:
            m.deltas = pair_deltas(m.a, m.b, alignment)
        if m.b_index is not None:
            m.geometry_anomaly = geometry_anomaly(m.b, same_line_neighbors(graph_b, m.b_index))

        if m.status == WordStatus.MODIFIED:
            m.suspicion = 0.6 + 0.25 * critical + 0.15 * m.geometry_anomaly
        elif m.status in (WordStatus.ADDED, WordStatus.REMOVED):
            m.suspicion = 0.5 + 0.25 * critical
        else:  # MOVED
            m.suspicion = 0.2


def document_anomalies(graph: nx.Graph, threshold: float = 0.5, top: int = 5) -> list[AnomalyWord]:
    scored = []
    for n in graph.nodes:
        word = graph.nodes[n]["word"]
        scored.append(AnomalyWord(text=word.text, score=geometry_anomaly(word, same_line_neighbors(graph, n)), index=n))
    return sorted((a for a in scored if a.score >= threshold), key=lambda a: a.score, reverse=True)[:top]
