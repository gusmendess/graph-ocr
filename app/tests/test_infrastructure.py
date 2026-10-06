import numpy as np
import pytest

from app.api.domain.bbox import BBox
from app.api.domain.entities import Word
from app.api.infrastructure.image_files import expand_images
from app.api.infrastructure.image_io import decode_image
from app.api.infrastructure.output_io import write_output
from app.api.repositories.ocr_cache import OcrCacheRepository
from app.api.services import hashing_service as hs
from app.api.services.ocr_export_service import render_ocr_text
from app.api.services.ocr_lines import group_lines
from app.tests.conftest import make_document, png


def test_bbox_iou():
    a = BBox(0, 0, 10, 10)
    assert a.iou(a) == 1.0
    assert a.iou(BBox(20, 20, 30, 30)) == 0.0
    assert a.iou(BBox(5, 0, 15, 10)) == pytest.approx(1 / 3)


def test_expand_images_lists_folders_alphabetically_and_ignores_other_files(tmp_path):
    (tmp_path / "b.png").write_bytes(b"x")
    (tmp_path / "a.JPG").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / "sub").mkdir()
    assert [p.name for p in expand_images([tmp_path])] == ["a.JPG", "b.png"]
    assert [p.name for p in expand_images([tmp_path / "b.png", tmp_path / "a.JPG"])] == ["b.png", "a.JPG"]  # na ordem passada


def test_expand_images_fails_loudly_for_a_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        expand_images([tmp_path / "nao_existe"])


def test_write_output_falls_back_when_the_target_cannot_be_written(tmp_path, capsys):
    target = tmp_path / "saida.png"
    target.mkdir()  # um diretório com o mesmo nome: a escrita falha como se o arquivo estivesse bloqueado
    written = write_output(target, b"dados", retries=1, delay=0)
    assert written != target and written.read_bytes() == b"dados"
    assert "aviso" in capsys.readouterr().err


def test_write_output_writes_text_and_bytes(tmp_path):
    assert write_output(tmp_path / "a.txt", "olá").read_text(encoding="utf-8") == "olá"
    assert write_output(tmp_path / "b.bin", b"\x00\x01").read_bytes() == b"\x00\x01"


def test_decode_image_rejects_garbage():
    with pytest.raises(ValueError):
        decode_image(b"isto nao e uma imagem")


def test_sha_and_phash_identity():
    image = make_document(1)
    assert hs.sha256_bytes(png(image)) == hs.sha256_bytes(png(image))
    assert hs.sha256_bytes(png(make_document(1))) != hs.sha256_bytes(png(make_document(2)))  # 1 pixel muda o sha256
    bits = hs.phash(image)
    assert bits.shape == (64,) and hs.phash_similarity(bits, hs.phash(make_document(2))) == 1.0  # mas não o phash


def test_phash_separates_very_different_pages_and_roundtrips_through_hex():
    a, b = hs.phash(make_document(1)), hs.phash(make_document(1, inverted=True))
    assert hs.phash_similarity(a, b) < 0.5
    assert np.array_equal(hs.phash_from_hex(hs.phash_to_hex(a)), a)
    assert len(hs.phash_to_hex(a)) == 16


def test_phash_thumbnail_is_32x32_gray():
    assert hs.phash_thumbnail(make_document(1)).shape == (32, 32)


def test_group_lines_orders_words_inside_and_lines_top_to_bottom():
    words = [
        Word("mundo", BBox(60, 40, 100, 60), 0.9, 7),
        Word("ola", BBox(10, 40, 50, 60), 0.8, 7),
        Word("topo", BBox(10, 5, 50, 25), 1.0, 3),
    ]
    lines = group_lines(words)
    assert [ln.word.text for ln in lines] == ["topo", "ola mundo"]
    assert lines[1].word.conf == pytest.approx(0.85)


def test_group_lines_without_line_id_makes_each_word_a_line():
    words = [Word("a", BBox(0, 0, 5, 5), 1.0), Word("b", BBox(0, 10, 5, 15), 1.0)]
    assert len(group_lines(words)) == 2


def test_ocr_text_export_lists_lines_and_words():
    words = [Word("Valor", BBox(10, 10, 60, 30), 0.99, 0), Word("R$", BBox(10, 50, 40, 70), 0.97, 1), Word("37,00", BBox(50, 50, 100, 70), 0.98, 1)]
    text = render_ocr_text(words, "x.png", "rapidocr")
    assert "# motor: rapidocr | linhas: 2 | palavras: 3" in text
    assert "| R$ 37,00" in text
    assert "| 37,00" in text.split("## Palavras")[1]


def test_ocr_cache_roundtrip_and_miss(tmp_path):
    cache = OcrCacheRepository(tmp_path)
    words = [Word("a", BBox(1, 2, 3, 4), 0.5, 0), Word("b", BBox(5, 6, 7, 8), 0.7, None)]
    assert cache.get("sha", "rapidocr") is None
    cache.put("sha", "rapidocr", words)
    assert cache.get("sha", "rapidocr") == words
    assert cache.get("sha", "tesseract") is None  # a chave inclui o motor
