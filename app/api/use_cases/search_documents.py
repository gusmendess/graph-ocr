from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.api.domain.entities import Word
from app.api.domain.vectors import DocumentVector
from app.api.infrastructure.image_io import decode_image
from app.api.models.schemas import IndexResponse, SearchHit, SearchResponse
from app.api.repositories.index_repository import DocumentIndexRepository
from app.api.services import hashing_service
from app.api.services.document_embedder import DocumentEmbedder
from app.api.services.ocr_service import OcrService
from app.api.services.vector_index import NumpyFlatIndex
from app.api.settings import Settings
from app.api.utils import sha256_bytes

_STAGE_ORDER = {"sha256": 0, "vector": 1, "phash": 2}  # quem passou pelos portões vem antes de quem foi barrado


@dataclass
class SearchOutcome:
    response: SearchResponse
    sha: str
    query: DocumentVector | None  # None se nenhum par chegou ao estágio de vetor
    words: list[Word] | None  # OCR da consulta (None se o OCR não rodou)


@dataclass
class _Candidate:
    pos: int
    score: float
    decided_by: str
    phash: float | None
    block_scores: dict[str, float]


class SearchDocumentsUseCase:
    """Busca reversa em cascata, de barata a cara. Cada par (consulta, documento indexado) para no 1º estágio que decide:

    1. sha256: bytes idênticos -> score 1.0 (~0 ms).
    2. phash:  similaridade perceptual abaixo de `phash_threshold` -> o próprio phash decide (score = a similaridade),
               sem OCR e sem rede neural.
    3. vetor:  só para os pares ainda ambíguos: cosseno do vetor multimodal (visual + texto + layout), de 0 a 1.

    O vetor da consulta (OCR + DINOv2) é calculado uma única vez, e só se algum par chegar ao estágio 3.
    Sem pareamento por campo, sem grafo e sem veredito de adulteração.
    """

    def __init__(
        self,
        settings: Settings,
        ocr: OcrService,
        embedder: DocumentEmbedder,
        repository: DocumentIndexRepository,
        index: NumpyFlatIndex | None = None,
    ) -> None:
        self._settings = settings
        self._ocr = ocr
        self._embedder = embedder
        self._repository = repository
        self._index = index if index is not None else repository.load()

    @property
    def n_indexed(self) -> int:
        return len(self._index)

    @property
    def index(self) -> NumpyFlatIndex:
        return self._index

    @property
    def engine_name(self) -> str:
        return self._ocr.engine_name

    def document_path(self, doc_id: str) -> Path:
        """Cópia do documento original guardada junto do índice."""
        return Path(self._index.metas[self._index.ids.index(doc_id)]["path"])

    def _embed(self, image: np.ndarray, sha: str) -> tuple[DocumentVector, list[Word]]:
        words = self._ocr.read(sha, image)
        return self._embedder.embed(image, words), words

    def add(self, data: bytes, label: str, persist: bool = True) -> IndexResponse:
        sha = sha256_bytes(data)
        replaced = sha in self._index.ids
        image = decode_image(data)
        vector, _ = self._embed(image, sha)
        path = self._repository.save_document(sha, data, Path(label).suffix or ".png")
        meta = {"label": label, "path": str(path), "phash": hashing_service.phash_to_hex(hashing_service.phash(image))}
        self._index.add(sha, vector, meta)
        if persist:
            self.save()
        return IndexResponse(
            doc_id=sha, label=label, replaced=replaced, n_indexed=len(self._index), blocks=sorted(vector.blocks)
        )

    def save(self) -> None:
        self._repository.save(self._index)

    def _phash_of(self, pos: int) -> np.ndarray:
        """phash do documento indexado; entradas antigas (sem phash guardado) são calculadas da cópia no índice."""
        meta = self._index.metas[pos]
        if "phash" not in meta:
            meta["phash"] = hashing_service.phash_to_hex(
                hashing_service.phash(decode_image(Path(meta["path"]).read_bytes()))
            )
        return hashing_service.phash_from_hex(meta["phash"])

    def search(self, data: bytes, k: int | None = None, weights: dict[str, float] | None = None) -> SearchOutcome:
        settings = self._settings
        sha = sha256_bytes(data)
        image = decode_image(data)
        query_phash = hashing_service.phash(image)
        weights = weights or settings.search_weights()

        candidates: list[_Candidate] = []
        pending: dict[int, float] = {}  # posição -> similaridade phash, dos pares que seguem para o vetor
        for pos, doc_id in enumerate(self._index.ids):
            if doc_id == sha:
                candidates.append(_Candidate(pos, 1.0, "sha256", None, {}))
                continue
            similarity = hashing_service.phash_similarity(query_phash, self._phash_of(pos))
            if similarity < settings.phash_threshold:
                candidates.append(_Candidate(pos, similarity, "phash", similarity, {}))
            else:
                pending[pos] = similarity

        query = words = None
        weights_used: dict[str, float] = {}
        if pending:  # estágio 3: o único que custa OCR + rede neural
            query, words = self._embed(image, sha)
            total, per_block, weights_used = self._index.score_all(query, weights)
            for pos, similarity in pending.items():
                blocks = {b: float(per_block[b][pos]) for b in sorted(per_block)}
                candidates.append(_Candidate(pos, float(total[pos]), "vector", similarity, blocks))

        candidates.sort(key=lambda c: (_STAGE_ORDER[c.decided_by], -c.score))
        hits = [
            SearchHit(
                doc_id=self._index.ids[c.pos],
                label=self._index.metas[c.pos].get("label", self._index.ids[c.pos]),
                score=c.score,
                decided_by=c.decided_by,
                phash=c.phash,
                block_scores=c.block_scores,
                is_query=self._index.ids[c.pos] == sha,
            )
            for c in candidates[: k or settings.search_k]
        ]
        response = SearchResponse(
            exact_match=sha in self._index.ids,
            vector_computed=query is not None,
            phash_threshold=settings.phash_threshold,
            stage_counts=dict(Counter(c.decided_by for c in candidates)),
            blocks_used=sorted(weights_used),
            weights_used=weights_used,
            n_indexed=len(self._index),
            hits=hits,
        )
        return SearchOutcome(response, sha, query, words)
