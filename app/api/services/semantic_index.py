from dataclasses import dataclass, field

import numpy as np


@dataclass
class SemanticHit:
    doc_id: str
    score: float  # cosseno entre o vetor da consulta e o da imagem
    meta: dict = field(default_factory=dict)


class SemanticIndex:
    """Índice exato de vetores de imagem (um por imagem). Consulta: vetor de texto (ou de outra imagem).

    O cosseno do SigLIP entre texto e imagem fica em valores baixos (algo como 0.05 a 0.35) mesmo para acertos:
    vale pela ORDEM do ranking, não como probabilidade.
    """

    def __init__(self) -> None:
        self.ids: list[str] = []
        self.metas: list[dict] = []
        self.matrix: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self.model: str | None = None  # modelo + pré-processamento que geraram os vetores (não se misturam)

    def __len__(self) -> int:
        return len(self.ids)

    def add(self, doc_id: str, vector: np.ndarray, meta: dict | None = None) -> None:
        vector = vector.astype(np.float32)[None]
        if doc_id in self.ids:  # mesma imagem (mesmos bytes): substitui
            pos = self.ids.index(doc_id)
            self.matrix[pos] = vector[0]
            self.metas[pos] = meta or {}
            return
        self.matrix = vector if len(self) == 0 else np.vstack([self.matrix, vector])
        self.ids.append(doc_id)
        self.metas.append(meta or {})

    def search(self, query: np.ndarray, k: int) -> list[SemanticHit]:
        if not self.ids:
            return []
        scores = self.matrix @ query.astype(np.float32)
        order = np.argsort(-scores)[:k]
        return [SemanticHit(self.ids[i], float(scores[i]), self.metas[i]) for i in order]
