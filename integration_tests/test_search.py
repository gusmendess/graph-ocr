from pathlib import Path

import pytest

from app.api.adapters.ocr.rapidocr_engine import RapidOcrEngine
from app.api.repositories.index_repository import DocumentIndexRepository
from app.api.services.embedder_factory import build_document_embedder
from app.api.services.ocr_service import OcrService
from app.api.settings import Settings
from app.api.use_cases.search_documents import SearchDocumentsUseCase

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"
INDEXED = ["original.png", "fraud_day.png", "fraud_day_time.png", "image4.png", "image7.png"]
TEXT_ONLY = {"visual": 0.0, "text": 1.0, "layout": 0.0}


@pytest.fixture(scope="module")
def use_case(tmp_path_factory):
    root = tmp_path_factory.mktemp("index")
    # não baixa modelo no teste: usa o bloco visual só se o ONNX já estiver em cache
    settings = Settings(index_dir=root, cache_dir=root / "ocr", visual_auto_download=False)
    search = SearchDocumentsUseCase(
        settings, OcrService(RapidOcrEngine()), build_document_embedder(settings), DocumentIndexRepository(root)
    )
    for name in INDEXED:
        search.add((SAMPLES / name).read_bytes(), name, persist=False)
    search.save()
    return search


def test_same_file_scores_one_and_edited_receipts_score_just_below(use_case):
    response = use_case.search((SAMPLES / "fraud_day.png").read_bytes(), k=5).response
    by_label = {h.label: h for h in response.hits}
    assert response.exact_match and by_label["fraud_day.png"].score == pytest.approx(1.0)
    assert by_label["original.png"].score < 1.0  # a data editada é a única diferença
    assert by_label["original.png"].score > 0.95  # mas o cosseno é altíssimo: o vetor mede semelhança, não adulteração
    assert {"text", "layout"} <= set(response.blocks_used)


def test_every_score_is_between_0_and_1(use_case):
    response = use_case.search((SAMPLES / "image6.jpeg").read_bytes(), k=10).response
    assert response.hits
    for hit in response.hits:
        assert 0.0 <= hit.score <= 1.0
        assert all(0.0 <= v <= 1.0 for v in hit.block_scores.values())


def test_photo_of_an_edited_screen_finds_its_source_by_content(use_case):
    """image6.jpeg é a foto da tela do image4: por texto, o image4 é o par mais parecido."""
    hits = use_case.search((SAMPLES / "image6.jpeg").read_bytes(), k=5, weights=TEXT_ONLY).response.hits
    assert hits[0].label == "image4.png"
    assert hits[0].score > hits[-1].score
