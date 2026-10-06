from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from app.api.domain.bbox import BBox
from app.api.domain.entities import Word


@dataclass
class Line:
    indices: list[int]  # índices das palavras (na lista do OCR) que formam a linha
    word: Word  # linha como "palavra" única: texto, bbox união, confiança média


def group_lines(words: list[Word]) -> list[Line]:
    """Agrupa palavras por linha do OCR, em ordem de leitura. Sem line_id, cada palavra é uma linha."""
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
