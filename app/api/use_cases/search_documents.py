from dataclasses import dataclass
from pathlib import Path

from app.api.domain.vectors import DocumentVector, normalize_weights
from app.api.infrastructure.image_io import decode_image
from app.api.models.schemas import CascadeResult, IndexResponse, SearchHit, SearchResponse
from app.api.repositories.index_repository import DocumentIndexRepository
from app.api.services.document_embedder import DocumentEmbedder
from app.api.services.ocr_service import OcrService
from app.api.services.vector_index import Hit, NumpyFlatIndex
from app.api.settings import Settings
from app.api.use_cases.compare_documents import CompareDocumentsUseCase
from app.api.utils import sha256_bytes


@dataclass
class SearchOutcome:
    response: SearchResponse
    hits: list[Hit]


class SearchDocumentsUseCase:
    """Busca reversa por vetor multimodal: acha os documentos indexados mais parecidos com a consulta.

    Posição na cascata: sha256 -> busca vetorial top-k -> OCR + grafo nos k candidatos. O vetor só
    RECUPERA candidatos (original e fraude têm cosseno altíssimo); o veredito de adulteração continua
    vindo do pareamento por campo, em `search_and_compare`.
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

    def search_and_compare(
        self, data: bytes, compare: CompareDocumentsUseCase, k: int | None = None, weights: dict[str, float] | None = None
    ) -> list[tuple[SearchHit, CascadeResult | None]]:
        """Busca vetorial + cascata de adulteração (OCR + grafo) em cada candidato (a consulta em si fica de fora)."""
        outcome = self.search(data, k, weights)
        results: list[tuple[SearchHit, CascadeResult | None]] = []
        for hit, raw in zip(outcome.response.hits, outcome.hits):
            if hit.is_query:
                results.append((hit, None))
                continue
            candidate = Path(raw.meta["path"]).read_bytes()
            results.append((hit, compare.execute(candidate, data, hit.label, "consulta", run_all=True).result))
        return results
