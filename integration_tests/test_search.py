from pathlib import Path

import pytest

from app.api.adapters.ocr.rapidocr_engine import RapidOcrEngine
from app.api.domain.enums import FieldKind, WordStatus
from app.api.repositories.index_repository import DocumentIndexRepository
from app.api.services.embedder_factory import build_document_embedder
from app.api.services.ocr_service import OcrService
from app.api.settings import Settings
from app.api.use_cases.compare_documents import CompareDocumentsUseCase
from app.api.use_cases.search_documents import SearchDocumentsUseCase

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"
INDEXED = ["original.png", "fraud_day.png", "fraud_day_time.png", "image4.png", "image7.png"]
TEXT_ONLY = {"visual": 0.0, "text": 1.0, "layout": 0.0}


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    root = tmp_path_factory.mktemp("index")
    # não baixa modelo no teste: usa o bloco visual só se o ONNX já estiver em cache
    settings = Settings(index_dir=root, cache_dir=root / "ocr", visual_auto_download=False)
    ocr = OcrService(RapidOcrEngine())
    use_case = SearchDocumentsUseCase(settings, ocr, build_document_embedder(settings), DocumentIndexRepository(root))
    for name in INDEXED:
        use_case.add((SAMPLES / name).read_bytes(), name, persist=False)
    use_case.save()
    return use_case, CompareDocumentsUseCase(settings, ocr)


def test_edited_receipts_retrieve_the_same_family(stack):
    use_case, _ = stack
    response = use_case.search((SAMPLES / "fraud_day.png").read_bytes(), k=5).response
    others = [h for h in response.hits if not h.is_query]
    assert response.exact_match
    assert others[0].score > 0.95  # original/fraudes: cosseno altíssimo (por isso o vetor só recupera candidatos)
    assert {"text", "layout"} <= set(response.blocks_used)


def test_photo_of_an_edited_screen_finds_its_source_by_content(stack):
    """image6.jpeg é a foto da tela do image4: por texto, o image4 vem primeiro entre os não idênticos."""
    use_case, _ = stack
    data = (SAMPLES / "image6.jpeg").read_bytes()
    hits = use_case.search(data, k=5, weights=TEXT_ONLY).response.hits
    assert hits[0].label == "image4.png"


def test_search_then_compare_confirms_which_candidate_is_the_source(stack):
    """Vetor recupera, OCR + grafo decide: contra o image4 só sobra erro de leitura; contra o original, a data e o ID."""
    use_case, compare = stack
    rows = use_case.search_and_compare((SAMPLES / "image6.jpeg").read_bytes(), compare, k=5, weights=TEXT_ONLY)
    edits = {}
    for hit, result in rows:
        assert result is not None  # a foto não está no índice
        edits[hit.label] = {
            m.kind for m in result.matches if m.status == WordStatus.MODIFIED and m.kind in (FieldKind.DATE, FieldKind.ID)
        }
    assert edits["image4.png"] == set()
    assert edits["original.png"] == {FieldKind.DATE, FieldKind.ID}
