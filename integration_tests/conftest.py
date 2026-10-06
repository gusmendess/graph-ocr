"""Testes com os modelos reais (RapidOCR, DINOv2, SigLIP2) sobre as imagens de data/samples.

Não baixam nada: se os modelos não estiverem em .cache/models, os testes são pulados.
Rodam com `make test-integration` (ou `pytest -m integration`).
"""

from pathlib import Path

import pytest

from app.api.adapters.ocr.factory import build_engine
from app.api.repositories.index_repository import DocumentIndexRepository
from app.api.repositories.semantic_index_repository import SemanticIndexRepository
from app.api.services.embedder_factory import build_document_embedder, build_siglip_embedder
from app.api.services.ocr_service import OcrService
from app.api.services.siglip_presets import SIGLIP_PRESETS
from app.api.settings import Settings
from app.api.infrastructure.image_files import expand_images
from app.api.use_cases.search_documents import SearchDocumentsUseCase
from app.api.use_cases.search_images import SearchImagesUseCase

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"


def pytest_collection_modifyitems(items):
    for item in items:
        if Path(item.fspath).parent.name == "integration_tests":
            item.add_marker(pytest.mark.integration)


def _require(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"modelo ausente: {path} (rode o pipeline uma vez para baixá-lo)")


@pytest.fixture(scope="session")
def document_files() -> list[Path]:
    files = expand_images([SAMPLES / "documents"])
    if len(files) < 2:
        pytest.skip("precisa de pelo menos 2 imagens em data/samples/documents")
    return files


@pytest.fixture(scope="session")
def dress_files() -> list[Path]:
    files = expand_images([SAMPLES / "dresses"])
    if len(files) < 2:
        pytest.skip("precisa de imagens em data/samples/dresses")
    return files


@pytest.fixture(scope="session")
def documents_use_case(tmp_path_factory, document_files) -> SearchDocumentsUseCase:
    root = tmp_path_factory.mktemp("docs")
    settings = Settings(
        index_dir=root / "index", cache_dir=root / "ocr", ocr_engine="rapidocr", visual_auto_download=False,
        phash_threshold=0.0,  # todos os pares passam pelo vetor (o portão do phash é coberto nos testes unitários)
    )
    _require(settings.visual_model_path)
    use_case = SearchDocumentsUseCase(
        settings,
        OcrService(build_engine(settings)),
        build_document_embedder(settings),
        DocumentIndexRepository(settings.index_dir),
    )
    for file in document_files:
        use_case.add(file.read_bytes(), file.name, persist=False)
    return use_case


@pytest.fixture(scope="session")
def images_use_case(tmp_path_factory, dress_files) -> SearchImagesUseCase:
    settings = Settings(semantic_index_dir=tmp_path_factory.mktemp("semantic"))
    preset = SIGLIP_PRESETS[settings.siglip_model]
    for file in (preset.vision, preset.text, "tokenizer.json", *preset.extra):
        _require(settings.siglip_models_dir / preset.folder / Path(file).name)
    use_case = SearchImagesUseCase(
        settings, build_siglip_embedder(settings), SemanticIndexRepository(settings.semantic_index_dir)
    )
    use_case.add_many([(f.read_bytes(), f.name) for f in dress_files])
    return use_case
