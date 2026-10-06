from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from app.api.domain.vectors import BLOCKS, DocumentVector, normalize_weights


@dataclass
class Hit:
    doc_id: str
    score: float  # similaridade de 0 a 1: média ponderada dos cossenos de cada bloco (cada um truncado em 0)
    block_scores: dict[str, float]
    meta: dict = field(default_factory=dict)


class VectorIndex(Protocol):
    """Contrato do índice. Hoje: busca exata em numpy; FAISS/HNSW/pgvector entram atrás desta interface."""

    def add(self, doc_id: str, vector: DocumentVector, meta: dict | None = None) -> None: ...

    def search(self, query: DocumentVector, k: int, weights: dict[str, float]) -> list[Hit]: ...

    def __len__(self) -> int: ...


class NumpyFlatIndex:
    """Busca exata por produto interno (= cosseno, pois os blocos são L2-normalizados).

    Guarda os blocos em matrizes separadas: o placar composto é calculado na hora com os pesos pedidos,
    então dá para reponderar ou buscar por um bloco só sem reindexar. Serve bem até dezenas de milhares
    de documentos; acima disso, troque por um índice aproximado atrás de `VectorIndex`.
    """

    def __init__(self) -> None:
        self.ids: list[str] = []
        self.metas: list[dict] = []
        self.matrices: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.ids)

    def add(self, doc_id: str, vector: DocumentVector, meta: dict | None = None) -> None:
        if doc_id in self.ids:  # mesmo documento: substitui
            self._remove(self.ids.index(doc_id))
        if self.ids and set(vector.blocks) != set(self.matrices):
            raise ValueError(f"Blocos do vetor {sorted(vector.blocks)} diferem dos do índice {sorted(self.matrices)}")
        for name, vec in vector.blocks.items():
            row = vec.astype(np.float32)[None]
            self.matrices[name] = row if name not in self.matrices else np.vstack([self.matrices[name], row])
        self.ids.append(doc_id)
        self.metas.append(meta or {})

    def _remove(self, pos: int) -> None:
        keep = [i for i in range(len(self.ids)) if i != pos]
        self.matrices = {n: m[keep] for n, m in self.matrices.items()}
        self.ids = [self.ids[i] for i in keep]
        self.metas = [self.metas[i] for i in keep]

    def score_all(
        self, query: DocumentVector, weights: dict[str, float]
    ) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float]]:
        """Score composto e por bloco da consulta contra TODOS os documentos, mais os pesos efetivamente usados."""
        shared = {b for b in BLOCKS if b in query.blocks and b in self.matrices}
        used = normalize_weights(weights, shared)
        # cosseno truncado em [0, 1]: o score final fica em 0..1 (cosseno negativo = nenhuma semelhança)
        per_block = {b: np.clip(self.matrices[b] @ query.blocks[b].astype(np.float32), 0.0, 1.0) for b in shared}
        return sum(used[b] * per_block[b] for b in used), per_block, used

    def search(self, query: DocumentVector, k: int, weights: dict[str, float]) -> list[Hit]:
        if not self.ids:
            return []
        total, per_block, _ = self.score_all(query, weights)
        shared = set(per_block)
        order = np.argsort(-total)[:k]
        return [
            Hit(self.ids[i], float(total[i]), {b: float(per_block[b][i]) for b in sorted(shared)}, self.metas[i])
            for i in order
        ]
