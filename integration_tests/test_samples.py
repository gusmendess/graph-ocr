from pathlib import Path

import pytest

from app.api.adapters.ocr.rapidocr_engine import RapidOcrEngine
from app.api.domain.enums import Verdict, WordStatus
from app.api.services.ocr_service import OcrService
from app.api.settings import Settings
from app.api.use_cases.compare_documents import CompareDocumentsUseCase

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"


@pytest.fixture(scope="module")
def use_case():
    return CompareDocumentsUseCase(Settings(), OcrService(RapidOcrEngine()))


def _run(use_case, name):
    return use_case.execute((SAMPLES / "original.png").read_bytes(), (SAMPLES / name).read_bytes()).result


def test_same_file_stops_at_sha256(use_case):
    data = (SAMPLES / "original.png").read_bytes()
    result = use_case.execute(data, data).result
    assert result.verdict == Verdict.IDENTICAL
    assert [s.name for s in result.stages] == ["sha256"]


def test_day_edit_is_found(use_case):
    result = _run(use_case, "fraud_day.png")
    modified = [m for m in result.matches if m.status == WordStatus.MODIFIED]
    assert result.verdict == Verdict.TAMPERED
    assert [(m.a.text, m.b.text) for m in modified] == [("06/06/2021", "26/06/2021")]


def test_day_and_time_edit_are_found(use_case):
    result = _run(use_case, "fraud_day_time.png")
    modified = {(m.a.text, m.b.text) for m in result.matches if m.status == WordStatus.MODIFIED}
    assert modified == {("06/06/2021", "26/06/2021"), ("20:46:52", "20:52:06")}
    assert {f.code for f in result.doc_b.flags} == {"PIX_DATE_MISMATCH", "PIX_TIME_MISMATCH"}


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_alignment_recovers_random_warp_of_a_real_receipt(use_case, seed):
    """Aplica rotação/escala/deslocamento sorteados ao original e confere a transformação estimada."""
    import cv2
    import numpy as np

    from app.api.infrastructure.image_io import encode_png

    data = (SAMPLES / "original.png").read_bytes()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    rng = np.random.default_rng(seed)
    scale, deg = rng.uniform(0.9, 1.1), rng.uniform(-3, 3)
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, scale)
    m[:, 2] += rng.uniform(-50, 50, 2)
    warped = cv2.warpAffine(img, m, (w, h), borderValue=(255, 255, 255), flags=cv2.INTER_CUBIC)

    al = use_case.execute(data, encode_png(warped), run_all=True).result.alignment

    assert al.converged and al.inliers >= 60
    assert abs(al.scale - 1 / scale) < 0.03  # B -> A é a inversa de A -> B
    assert abs(al.rotation_deg - deg) < 0.5


def _analyze(name):
    from app.api.use_cases.analyze_document import AnalyzeDocumentUseCase

    uc = AnalyzeDocumentUseCase(Settings(), OcrService(RapidOcrEngine()))
    return uc.execute((SAMPLES / name).read_bytes())


def test_single_image_original_has_no_evidence():
    from app.api.domain.enums import DocumentVerdict

    assert _analyze("original.png").report.verdict == DocumentVerdict.NO_EVIDENCE


@pytest.mark.parametrize("name", ["fraud_day.png", "fraud_day_time.png", "fraud_day_time.jpeg"])
def test_single_image_flags_the_edited_fields(name):
    from app.api.domain.enums import DocumentVerdict

    outcome = _analyze(name)
    assert outcome.report.verdict == DocumentVerdict.SUSPICIOUS
    date_flag = next(f for f in outcome.report.flags if f.code == "PIX_DATE_MISMATCH")
    assert outcome.words[date_flag.word_indexes[0]].text.startswith("26/06/2021")


def test_full_mode_lets_ocr_override_phash_on_same_content(use_case):
    """Mesmo comprovante em outro enquadramento: o phash cai abaixo do limiar, mas o conteúdo é o mesmo."""
    import cv2
    import numpy as np

    from app.api.infrastructure.image_io import encode_png

    data = (SAMPLES / "original.png").read_bytes()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    canvas = np.full((int(h * 0.8), int(w * 2.2), 3), 20, np.uint8)  # fundo escuro, proporção bem diferente
    small = cv2.resize(img, (int(w * 0.6), int(h * 0.6)), interpolation=cv2.INTER_AREA)
    canvas[40 : 40 + small.shape[0], int(w * 0.8) : int(w * 0.8) + small.shape[1]] = small
    other = encode_png(canvas)

    fast = use_case.execute(data, other).result
    assert fast.verdict == Verdict.DIFFERENT_DOCUMENT and fast.decided_by == "phash"  # caminho rápido: phash decide

    full = use_case.execute(data, other, run_all=True).result
    assert full.verdict != Verdict.DIFFERENT_DOCUMENT  # modo completo: o OCR prevalece
    assert full.decided_by.startswith("ocr:")


def test_vector_matcher_flags_every_identity_field_replaced_entirely():
    """O fraudador troca TAMBÉM o código da transação, o ID Pix, a chave e o identificador por inteiro."""
    import random
    import string

    from app.api.domain.entities import Word
    from app.api.infrastructure.image_io import decode_image
    from app.api.services import vector_matching_service as vm
    from app.api.services.embedding_service import CharNgramEmbedder
    from app.api.utils import sha256_bytes

    data = (SAMPLES / "original.png").read_bytes()
    words = OcrService(RapidOcrEngine()).read(sha256_bytes(data), decode_image(data))
    rng = random.Random(0)
    ids = [i for i, w in enumerate(words) if len(w.text) >= 20 and sum(c.isdigit() for c in w.text) >= 6]
    assert len(ids) >= 4  # identificador, chave (1ª parte), código da transação, ID Pix

    def scramble(text):  # troca o valor inteiro preservando só o formato
        return "".join(rng.choice(string.digits) if c.isdigit() else rng.choice(string.ascii_uppercase) if c.isalpha() else c for c in text)

    tampered = [Word(scramble(w.text), w.bbox, w.conf, w.line_id) if i in ids else w for i, w in enumerate(words)]
    matches = vm.match_by_vectors(words, tampered, CharNgramEmbedder())

    modified = [m for m in matches if m.status == WordStatus.MODIFIED]
    assert {m.a_index for m in modified} == set(ids)
    assert all(m.status in (WordStatus.MATCHED, WordStatus.MODIFIED) for m in matches)
