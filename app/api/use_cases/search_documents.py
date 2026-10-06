from dataclasses import dataclass
from pathlib import Path

from app.api.domain.vectors import DocumentVector, normalize_weights
from app.api.infrastructure.image_io import decode_image
from app.api.models.schemas import IndexResponse, SearchHit, SearchResponse
from app.api.repositories.index_repository import DocumentIndexRepository
from app.api.services.document_embedder import DocumentEmbedder
from app.api.services.ocr_service import OcrService
from app.api.services.vector_index import Hit, NumpyFlatIndex
from app.api.settings import Settings
from app.api.utils import sha256_bytes


@dataclass
class SearchOutcome:
    response: SearchResponse
    hits: list[Hit]


class SearchDocumentsUseCase:
    """Busca reversa por vetor multimodal (visual + texto + layout) e cosseno.

    Devolve, para cada documento indexado parecido com a consulta, um score de 0 a 1. Só vetor + cosseno:
    sem pareamento por campo, sem grafo, sem veredito de adulteração. O OCR roda apenas para alimentar
    os blocos de texto e de layout do vetor.
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

    def _vector(self, data: bytes, sha: str) -> DocumentVector:
        image = decode_image(data)
        return self._embedder.embed(image, self._ocr.read(sha, image))

    def add(self, data: bytes, label: str, persist: bool = True) -> IndexResponse:
        sha = sha256_bytes(data)
        replaced = sha in self._index.ids
        vector = self._vector(data, sha)
        path = self._repository.save_document(sha, data, Path(label).suffix or ".png")
        self._index.add(sha, vector, {"label": label, "path": str(path)})
        if persist:
            self.save()
        return IndexResponse(
            doc_id=sha, label=label, replaced=replaced, n_indexed=len(self._index), blocks=sorted(vector.blocks)
        )

    def save(self) -> None:
        self._repository.save(self._index)

    def search(self, data: bytes, k: int | None = None, weights: dict[str, float] | None = None) -> SearchOutcome:
        sha = sha256_bytes(data)
        query = self._vector(data, sha)
        weights = weights or self._settings.search_weights()
        hits = self._index.search(query, k or self._settings.search_k, weights)
        shared = set(query.blocks) & set(self._index.matrices)
        response = SearchResponse(
            exact_match=sha in self._index.ids,
            blocks_used=sorted(shared),
            weights_used=normalize_weights(weights, shared) if shared else {},
            n_indexed=len(self._index),
            hits=[
                SearchHit(
                    doc_id=h.doc_id,
                    label=h.meta.get("label", h.doc_id),
                    score=h.score,
                    block_scores=h.block_scores,
                    is_query=h.doc_id == sha,
                )
                for h in hits
            ],
        )
        return SearchOutcome(response, hits)
