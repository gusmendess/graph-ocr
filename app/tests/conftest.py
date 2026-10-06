"""Auxiliares dos testes unitários: imagens sintéticas e dublês (OCR e embedder) que dispensam modelos e rede."""

import cv2
import numpy as np
import pytest

from app.api.domain.bbox import BBox
from app.api.domain.entities import Word
from app.api.infrastructure.image_io import encode_png
from app.api.repositories.index_repository import DocumentIndexRepository
from app.api.repositories.semantic_index_repository import SemanticIndexRepository
from app.api.services.document_embedder import DocumentEmbedder
from app.api.services.embedding_service import CharNgramEmbedder
from app.api.services.ocr_service import OcrService
from app.api.settings import Settings
from app.api.use_cases.search_documents import SearchDocumentsUseCase
from app.api.use_cases.search_images import SearchImagesUseCase


def make_document(tag: int, inverted: bool = False) -> np.ndarray:
    """Página de 'comprovante' sintética. `tag` vai no pixel (0, 0): muda os bytes sem mudar a aparência."""
    img = np.full((300, 200, 3), 15 if inverted else 255, np.uint8)
    for k in range(6):
        y = 20 + 40 * k
        cv2.rectangle(img, (20, y), (20 + (60 if inverted else 150), y + 14), (255, 255, 255) if inverted else (60, 60, 60), -1)
    img[0, 0, 0] = tag % 256
    return img


def png(image: np.ndarray) -> bytes:
    return encode_png(image)


class FakeOcrEngine:
    """OCR de mentira: gera 6 palavras (em 3 linhas) a partir da `tag` guardada no pixel (0, 0). Conta as chamadas."""

    name = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def read(self, image: np.ndarray) -> list[Word]:
        self.calls += 1
        tag = int(image[0, 0, 0])
        return [
            Word(f"t{tag}_{i}", BBox(10 + 60 * (i % 3), 10 + 30 * (i // 3), 60 + 60 * (i % 3), 30 + 30 * (i // 3)), 0.99, i // 3)
            for i in range(6)
        ]


COLORS = {"red": (0, 0, 255), "green": (0, 200, 0), "blue": (255, 0, 0), "yellow": (0, 230, 255)}  # BGR


def make_dress(color: str, shade: int = 0) -> np.ndarray:
    """'Vestido' sintético de cor conhecida (um polígono sobre fundo neutro). `shade` varia o tom sem mudar a cor."""
    img = np.full((220, 160, 3), (225, 225, 225), np.uint8)
    bgr = np.clip(np.array(COLORS[color], int) * (1 - 0.1 * shade) + 12 * shade, 0, 255).astype(np.uint8)
    cv2.fillPoly(img, [np.array([[60, 40], [100, 40], [96, 100], [140, 200], [20, 200], [64, 100]])], tuple(int(c) for c in bgr))
    return img


class ColorEmbedder:
    """Dublê do SigLIP: texto e imagem no mesmo espaço (3-D, RGB). Foto = cor média do vestido; texto = cor citada."""

    name = "fake-color"

    def embed_images(self, images: list[np.ndarray]) -> np.ndarray:
        out = []
        for image in images:
            mask = np.abs(image.astype(int) - 225).sum(axis=2) > 30  # tira o fundo
            rgb = image[mask].mean(axis=0)[::-1] if mask.any() else image.reshape(-1, 3).mean(axis=0)[::-1]
            out.append(rgb / np.linalg.norm(rgb))
        return np.array(out, dtype=np.float32)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        out = []
        for text in texts:
            rgb = np.array([sum(w in t for t in [text.lower()]) for w in ("red", "green", "blue")], float) + 1e-3
            out.append(rgb / np.linalg.norm(rgb))
        return np.array(out, dtype=np.float32)


@pytest.fixture
def ocr_engine() -> FakeOcrEngine:
    return FakeOcrEngine()


@pytest.fixture
def documents_use_case(tmp_path, ocr_engine):
    """Cascata sha256 -> phash -> vetor com OCR de mentira, sem bloco visual, sem rede e com limiar do phash ajustável."""

    def build(phash_threshold: float = 0.3) -> SearchDocumentsUseCase:
        settings = Settings(
            index_dir=tmp_path / "index", cache_dir=tmp_path / "ocr", visual_enabled=False, phash_threshold=phash_threshold
        )
        return SearchDocumentsUseCase(
            settings,
            OcrService(ocr_engine),
            DocumentEmbedder(CharNgramEmbedder(), visual=None),
            DocumentIndexRepository(settings.index_dir),
        )

    return build


@pytest.fixture
def images_use_case(tmp_path):
    settings = Settings(semantic_index_dir=tmp_path / "semantic")
    return SearchImagesUseCase(settings, ColorEmbedder(), SemanticIndexRepository(settings.semantic_index_dir))
