from dataclasses import dataclass
from pathlib import Path

from app.api.infrastructure.image_io import decode_image
from app.api.models.schemas import ImageHit, ImageIndexResponse, ImageSearchResponse
from app.api.repositories.semantic_index_repository import SemanticIndexRepository
from app.api.services.semantic_index import SemanticIndex
from app.api.services.text_image_embedder import SiglipEmbedder
from app.api.settings import Settings
from app.api.utils import sha256_bytes


@dataclass
class ImageSearchOutcome:
    response: ImageSearchResponse
    paths: list[Path]  # cópia local de cada imagem do ranking (para montar a folha de contato)


class SearchImagesUseCase:
    """Busca de imagens por linguagem natural: embeda as imagens uma vez e compara com o vetor da frase.

    Sem OCR, sem rótulos e sem legendas: o modelo (SigLIP2) já coloca texto e imagem no mesmo espaço.
    """

    def __init__(
        self,
        settings: Settings,
        embedder: SiglipEmbedder,
        repository: SemanticIndexRepository,
        index: SemanticIndex | None = None,
    ) -> None:
        self._settings = settings
        self._embedder = embedder
        self._repository = repository
        self._index = index if index is not None else repository.load()
        if len(self._index) and self._index.model != embedder.name:
            raise ValueError(
                f"O índice de imagens foi criado com '{self._index.model or 'modelo antigo (int8)'}' e o modelo atual é "
                f"'{embedder.name}': os vetores não são compatíveis. Apague {settings.semantic_index_dir} e rode "
                "`make index-images <pasta>` de novo."
            )
        self._index.model = embedder.name

    @property
    def n_indexed(self) -> int:
        return len(self._index)

    def add_many(self, items: list[tuple[bytes, str]], persist: bool = True) -> list[ImageIndexResponse]:
        """Indexa várias imagens (data, rótulo) de uma vez; os vetores são calculados em lote."""
        images = [decode_image(data) for data, _ in items]
        vectors = self._embedder.embed_images(images)
        out = []
        for (data, label), vector in zip(items, vectors):
            sha = sha256_bytes(data)
            replaced = sha in self._index.ids
            path = self._repository.save_image(sha, data, Path(label).suffix or ".png")
            self._index.add(sha, vector, {"label": label, "path": str(path)})
            out.append(ImageIndexResponse(doc_id=sha, label=label, replaced=replaced, n_indexed=len(self._index)))
        if persist:
            self._repository.save(self._index)
        return out

    def add(self, data: bytes, label: str, persist: bool = True) -> ImageIndexResponse:
        return self.add_many([(data, label)], persist)[0]

    def search_text(self, query: str, k: int | None = None) -> ImageSearchOutcome:
        if not query.strip():
            raise ValueError("A consulta está vazia")
        hits = self._index.search(self._embedder.embed_texts([query])[0], k or self._settings.search_k)
        response = ImageSearchResponse(
            query=query,
            n_indexed=len(self._index),
            hits=[
                ImageHit(rank=n, doc_id=h.doc_id, label=h.meta.get("label", h.doc_id), score=h.score)
                for n, h in enumerate(hits, 1)
            ],
        )
        return ImageSearchOutcome(response, [Path(h.meta["path"]) for h in hits])
