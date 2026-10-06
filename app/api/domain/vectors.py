from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

BLOCKS = ("visual", "text", "layout")


@dataclass
class DocumentVector:
    """Vetor multimodal de um documento: um bloco L2-normalizado por modalidade.

    Os blocos ficam separados de propósito: dá para reponderar, ou buscar só por texto ou só por
    aparência, sem reindexar. Um bloco ausente (ex.: sem modelo visual) simplesmente não participa.
    """

    blocks: dict[str, np.ndarray] = field(default_factory=dict)

    def composite(self, weights: dict[str, float]) -> np.ndarray:
        """[ sqrt(w_v)·visual | sqrt(w_t)·texto | sqrt(w_l)·layout ]: o produto interno de dois compostos
        é a média ponderada dos cossenos de cada bloco (útil para indexar num motor externo, ex.: FAISS)."""
        used = {b: w for b, w in normalize_weights(weights, set(self.blocks)).items() if w > 0}
        return np.concatenate([np.sqrt(w) * self.blocks[b] for b, w in used.items()])


def normalize_weights(weights: dict[str, float], available: set[str]) -> dict[str, float]:
    """Restringe aos blocos disponíveis e renormaliza para somar 1."""
    kept = {b: max(float(weights.get(b, 0.0)), 0.0) for b in BLOCKS if b in available}
    total = sum(kept.values())
    if total <= 0:
        raise ValueError("Nenhum bloco com peso positivo entre os blocos disponíveis")
    return {b: w / total for b, w in kept.items()}
