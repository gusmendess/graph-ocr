import numpy as np

from app.api.domain.bbox import BBox
from app.api.domain.entities import Word
from app.api.domain.enums import WordStatus
from app.api.services import entity_service, graph_service, hashing_service, matching_service


def _word(text, x, y, w=60, h=20, conf=0.99):
    return Word(text, BBox(x, y, x + w, y + h), conf)


def _match(words_a, words_b):
    ga, gb = graph_service.build_word_graph(words_a), graph_service.build_word_graph(words_b)
    return matching_service.match_words(words_a, words_b, ga, gb)


def test_iou():
    a = BBox(0, 0, 10, 10)
    assert a.iou(a) == 1.0
    assert a.iou(BBox(20, 20, 30, 30)) == 0.0
    assert abs(a.iou(BBox(5, 0, 15, 10)) - 1 / 3) < 1e-9


def test_phash_identical_and_different():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (200, 100, 3), dtype=np.uint8)
    other = rng.integers(0, 255, (200, 100, 3), dtype=np.uint8)
    assert hashing_service.phash_similarity(hashing_service.phash(img), hashing_service.phash(img)) == 1.0
    assert hashing_service.phash_similarity(hashing_service.phash(img), hashing_service.phash(other)) < 0.9


def test_same_slot_different_text_is_modified():
    base = [_word("Data", 10, 10), _word("06/06/2021", 100, 10, 100), _word("Valor", 10, 50)]
    edited = [_word("Data", 10, 10), _word("26/06/2021", 100, 10, 100), _word("Valor", 10, 50)]
    status = {m.status for m in _match(base, edited)}
    assert WordStatus.MODIFIED in status
    assert WordStatus.ADDED not in status and WordStatus.REMOVED not in status


def test_repeated_word_disambiguated_by_context():
    a = [_word("R$", 10, 10), _word("37,00", 80, 10), _word("R$", 10, 100), _word("40,00", 80, 100)]
    b = [_word("R$", 10, 100), _word("40,00", 80, 100), _word("R$", 10, 10), _word("37,00", 80, 10)]
    assert all(m.status == WordStatus.MATCHED for m in _match(a, b))


def test_removed_and_added():
    a = [_word("a", 10, 10), _word("b", 10, 100)]
    b = [_word("a", 10, 10), _word("zzz", 300, 300)]
    status = sorted(m.status.value for m in _match(a, b))
    assert status == ["added", "matched", "removed"]


def test_pix_timestamp_consistency():
    e2e = "E08561701202106062346JCZ9B94YX7F"  # 06/06/2021 23:46 UTC = 20:46 BRT
    ok = [_word("06/06/2021", 0, 0), _word("às", 0, 0), _word("20:46:52", 0, 0), _word(e2e, 0, 50)]
    bad = [_word("26/06/2021", 0, 0), _word("às", 0, 0), _word("20:52:06", 0, 0), _word(e2e, 0, 50)]
    assert entity_service.consistency_flags(ok) == []
    assert {f.code for f in entity_service.consistency_flags(bad)} == {"PIX_DATE_MISMATCH", "PIX_TIME_MISMATCH"}


def test_case_graph_shares_entities():
    from app.api.models.schemas import Entity

    shared = Entity(kind="pix_e2e", value="E1")
    graph = graph_service.build_case_graph({"a": [shared], "b": [shared, Entity(kind="cnpj", value="X")]})
    found = graph_service.shared_entities(graph)
    assert [(s.kind, s.documents) for s in found] == [("pix_e2e", ["a", "b"])]


def test_consistency_rule_reads_date_glued_to_text_and_points_to_the_right_words():
    """OCR de JPEG costuma colar tokens ("2021às"): a regra não pode cair na data de expiração."""
    e2e = "E08561701202106062346JCZ9B94YX7F"
    words = [
        _word("26/06/2021às", 0, 0),
        _word("20:46:52", 0, 20),
        _word("08/06/2021-20:45", 0, 80),  # expiração: não é a data do comprovante
        _word(e2e, 0, 120),
    ]
    flags = entity_service.consistency_flags(words)
    assert [f.code for f in flags] == ["PIX_DATE_MISMATCH"]
    assert flags[0].word_indexes == [0, 3]


def test_ocr_text_export_lists_lines_and_words():
    from app.api.services.ocr_export_service import render_ocr_text

    words = [
        Word("Valor", BBox(10, 10, 60, 30), 0.99, 0),
        Word("R$", BBox(10, 50, 40, 70), 0.97, 1),
        Word("37,00", BBox(50, 50, 100, 70), 0.98, 1),
    ]
    text = render_ocr_text(words, "x.png", "rapidocr")
    assert "# motor: rapidocr | linhas: 2 | palavras: 3" in text
    assert "| R$ 37,00" in text  # linha reconstruída
    assert "| 37,00" in text.split("## Palavras")[1]
