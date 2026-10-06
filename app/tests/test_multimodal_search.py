"""Busca por vetor multimodal: blocos, composição, índice e caso de uso (sem modelo visual nem OCR reais)."""

import cv2
import numpy as np
import pytest

from app.api.domain.bbox import BBox
from app.api.domain.entities import Word
from app.api.domain.vectors import DocumentVector, normalize_weights
from app.api.infrastructure.image_io import encode_png
from app.api.repositories.index_repository import DocumentIndexRepository
from app.api.services.document_embedder import DocumentEmbedder
from app.api.services.embedding_service import CharNgramEmbedder
from app.api.services.layout_embedder import layout_vector
from app.api.services.ocr_service import OcrService
from app.api.services.vector_index import NumpyFlatIndex
from app.api.settings import Settings
from app.api.use_cases.search_documents import SearchDocumentsUseCase


def _unit(rng: np.random.Generator, dim: int) -> np.ndarray:
    v = rng.normal(size=dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _vec(rng, visual=True) -> DocumentVector:
    blocks = {"text": _unit(rng, 64), "layout": _unit(rng, 16)}
    if visual:
        blocks["visual"] = _unit(rng, 32)
    return DocumentVector(blocks)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def _doc(seed: int, n: int = 30) -> list[Word]:
    rng = np.random.default_rng(seed)
    out, y = [], 20.0
    for i in range(n):
        x = float(rng.uniform(10, 60))
        w = float(rng.uniform(40, 200))
        h = float(rng.uniform(14, 26))
        out.append(Word(f"w{seed}_{i}", BBox(x, y, x + w, y + h), 0.99, i))
        y += h + float(rng.uniform(8, 30))
    return out


# ---- bloco de layout -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(4))
def test_layout_is_invariant_to_scale_and_translation(seed):
    words = _doc(seed)
    rng = np.random.default_rng(seed)
    s, dx, dy = float(rng.uniform(0.4, 2.5)), float(rng.uniform(-300, 300)), float(rng.uniform(-300, 300))
    moved = [Word(w.text, BBox(w.bbox.x0 * s + dx, w.bbox.y0 * s + dy, w.bbox.x1 * s + dx, w.bbox.y1 * s + dy), w.conf, w.line_id) for w in words]
    assert _cos(layout_vector(words), layout_vector(moved)) > 0.999


def test_layout_separates_different_arrangements_and_handles_empty():
    assert _cos(layout_vector(_doc(1)), layout_vector(_doc(2))) < 0.95
    assert not layout_vector([]).any()
    assert layout_vector(_doc(1)).shape == (256,)


# ---- composição ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(5))
def test_composite_dot_product_is_the_weighted_average_of_block_cosines(seed):
    rng = np.random.default_rng(seed)
    a, b = _vec(rng), _vec(rng)
    weights = {"visual": 0.5, "text": 0.3, "layout": 0.2}
    expected = sum(weights[n] * float(a.blocks[n] @ b.blocks[n]) for n in weights)
    assert float(a.composite(weights) @ b.composite(weights)) == pytest.approx(expected, abs=1e-5)


def test_weights_are_renormalized_over_available_blocks():
    assert normalize_weights({"visual": 0.4, "text": 0.4, "layout": 0.2}, {"text", "layout"}) == {
        "text": pytest.approx(2 / 3),
        "layout": pytest.approx(1 / 3),
    }
    with pytest.raises(ValueError):
        normalize_weights({"visual": 1.0}, {"text"})


# ---- índice ----------------------------------------------------------------------------------------------------

def test_index_returns_exact_match_first_with_per_block_scores():
    rng = np.random.default_rng(0)
    index, vectors = NumpyFlatIndex(), {}
    for name in "abcde":
        vectors[name] = _vec(rng)
        index.add(name, vectors[name], {"label": name})
    hits = index.search(vectors["c"], 3, {"visual": 0.4, "text": 0.4, "layout": 0.2})
    assert hits[0].doc_id == "c" and hits[0].score == pytest.approx(1.0, abs=1e-5)
    assert set(hits[0].block_scores) == {"visual", "text", "layout"}
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_reweighting_changes_the_ranking_without_reindexing():
    """O mesmo índice responde "só texto" ou "só visual": os blocos ficam separados."""
    index = NumpyFlatIndex()
    q = DocumentVector({"text": np.array([1, 0], np.float32), "visual": np.array([1, 0], np.float32)})
    index.add("igual_no_texto", DocumentVector({"text": np.array([1, 0], np.float32), "visual": np.array([0, 1], np.float32)}))
    index.add("igual_no_visual", DocumentVector({"text": np.array([0, 1], np.float32), "visual": np.array([1, 0], np.float32)}))
    assert index.search(q, 1, {"text": 1.0, "visual": 0.0})[0].doc_id == "igual_no_texto"
    assert index.search(q, 1, {"text": 0.0, "visual": 1.0})[0].doc_id == "igual_no_visual"


def test_index_replaces_same_id_and_rejects_mismatched_blocks():
    rng = np.random.default_rng(1)
    index = NumpyFlatIndex()
    index.add("x", _vec(rng))
    index.add("x", _vec(rng))
    assert len(index) == 1
    with pytest.raises(ValueError):
        index.add("y", _vec(rng, visual=False))


def test_search_uses_only_blocks_shared_by_query_and_index():
    rng = np.random.default_rng(2)
    index = NumpyFlatIndex()
    index.add("x", _vec(rng))  # indexado com visual
    hits = index.search(_vec(rng, visual=False), 1, {"visual": 0.4, "text": 0.4, "layout": 0.2})
    assert set(hits[0].block_scores) == {"text", "layout"}


def test_repository_roundtrip(tmp_path):
    rng = np.random.default_rng(3)
    index = NumpyFlatIndex()
    for name in ("a", "b"):
        index.add(name, _vec(rng), {"label": name, "path": f"/x/{name}.png"})
    repo = DocumentIndexRepository(tmp_path)
    repo.save(index)
    loaded = repo.load()
    assert loaded.ids == ["a", "b"] and loaded.metas == index.metas
    assert np.allclose(loaded.matrices["text"], index.matrices["text"])
    assert len(DocumentIndexRepository(tmp_path / "vazio").load()) == 0


# ---- caso de uso -----------------------------------------------------------------------------------------------

class _FakeEngine:
    """OCR de mentira: devolve as palavras de um documento sintético escolhido pela cor do pixel (0,0)."""

    name = "fake"

    def __init__(self) -> None:
        self.docs = {int(s): _doc(int(s)) for s in (10, 20, 30)}

    def read(self, image):
        return self.docs[int(image[0, 0, 0])]


def _image(tag: int) -> bytes:
    img = np.full((200, 100, 3), 255, np.uint8)
    img[0, 0, 0] = tag
    cv2.putText(img, str(tag), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return encode_png(img)


def test_use_case_indexes_searches_and_flags_exact_match(tmp_path):
    settings = Settings(index_dir=tmp_path, cache_dir=tmp_path / "ocr", visual_enabled=False)
    embedder = DocumentEmbedder(CharNgramEmbedder(), visual=None)
    use_case = SearchDocumentsUseCase(settings, OcrService(_FakeEngine()), embedder, DocumentIndexRepository(tmp_path))
    for tag in (10, 20):
        use_case.add(_image(tag), f"doc{tag}.png")
    assert use_case.n_indexed == 2

    outcome = use_case.search(_image(10), k=2)
    top = outcome.response
    assert top.exact_match and top.hits[0].label == "doc10.png" and top.hits[0].is_query
    assert top.blocks_used == ["layout", "text"]
    assert top.hits[0].score > top.hits[1].score

    unseen = use_case.search(_image(30), k=2).response  # documento que não está no índice
    assert not unseen.exact_match and not any(h.is_query for h in unseen.hits)

    reloaded = SearchDocumentsUseCase(settings, OcrService(_FakeEngine()), embedder, DocumentIndexRepository(tmp_path))
    assert reloaded.n_indexed == 2  # persistido em disco


def test_different_document_scores_far_below_variants_of_the_same_one():
    """Controle sintético (texto + layout): outro documento fica bem abaixo de uma edição do mesmo."""
    emb = DocumentEmbedder(CharNgramEmbedder(), visual=None)
    img = np.full((100, 100, 3), 255, np.uint8)
    base = _doc(1)
    edited = [Word("EDITADO" if i == 5 else w.text, w.bbox, w.conf, w.line_id) for i, w in enumerate(base)]
    index = NumpyFlatIndex()
    index.add("base", emb.embed(img, base))
    index.add("outro", emb.embed(img, _doc(2)))
    hits = {h.doc_id: h.score for h in index.search(emb.embed(img, edited), 2, {"text": 0.5, "layout": 0.5})}
    assert hits["base"] > 0.9 and hits["outro"] < hits["base"] - 0.3


def test_rank_references_prefers_content_then_cosine_and_drops_the_query():
    from types import SimpleNamespace as NS

    from app.api.use_cases.search_documents import rank_references

    def row(label, cosine, content, is_query=False):
        hit = NS(label=label, score=cosine, is_query=is_query)
        return hit, None if is_query else NS(score=content)

    rows = [
        row("a", 0.957, 0.950),
        row("consulta", 1.000, 0, is_query=True),
        row("b", 0.956, 0.974),  # cosseno um pouco menor, mas explica melhor o conteúdo
        row("c", 0.960, 0.974),  # empate no conteúdo: vence o maior cosseno
    ]
    assert [hit.label for hit, _ in rank_references(rows)] == ["c", "b", "a"]
    assert rank_references([row("consulta", 1.0, 0, is_query=True)]) == []
