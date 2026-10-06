import numpy as np
import pytest

from app.api.domain.bbox import BBox
from app.api.domain.entities import Word
from app.api.domain.vectors import DocumentVector, normalize_weights
from app.api.repositories.index_repository import DocumentIndexRepository
from app.api.services.embedding_service import CharNgramEmbedder, cosine_matrix
from app.api.services.layout_embedder import layout_vector
from app.api.services.vector_index import NumpyFlatIndex


def _unit(rng, dim):
    v = rng.normal(size=dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _vector(rng, visual=True) -> DocumentVector:
    blocks = {"text": _unit(rng, 64), "layout": _unit(rng, 16)}
    if visual:
        blocks["visual"] = _unit(rng, 32)
    return DocumentVector(blocks)


def _words(seed, n=25):
    rng = np.random.default_rng(seed)
    out, y = [], 20.0
    for i in range(n):
        x, w, h = float(rng.uniform(10, 60)), float(rng.uniform(40, 200)), float(rng.uniform(14, 26))
        out.append(Word(f"w{i}", BBox(x, y, x + w, y + h), 0.99, i))
        y += h + float(rng.uniform(8, 30))
    return out


# ---- bloco de texto ------------------------------------------------------------------------------------------

def test_char_ngram_keeps_digit_differences_visible_and_is_deterministic():
    emb = CharNgramEmbedder()
    same = cosine_matrix(emb.embed(["06/06/2021"]), emb.embed(["06/06/2021"]))[0, 0]
    near = cosine_matrix(emb.embed(["06/06/2021"]), emb.embed(["26/06/2021"]))[0, 0]
    far = cosine_matrix(emb.embed(["06/06/2021"]), emb.embed(["Chave"]))[0, 0]
    assert same == pytest.approx(1.0) and far < near < 1.0
    assert np.array_equal(emb.embed(["abc"]), emb.embed(["abc"]))


def test_char_ngram_ignores_case_accents_and_spacing():
    emb = CharNgramEmbedder()
    assert cosine_matrix(emb.embed(["Comprovante  de PAGAMENTO"]), emb.embed(["comprovante de pagamento"]))[0, 0] == pytest.approx(1.0)
    assert cosine_matrix(emb.embed(["Instituição"]), emb.embed(["instituicao"]))[0, 0] == pytest.approx(1.0)


# ---- bloco de layout -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(3))
def test_layout_is_invariant_to_scale_and_translation(seed):
    words = _words(seed)
    s, dx, dy = 1.7, 300.0, -80.0
    moved = [Word(w.text, BBox(w.bbox.x0 * s + dx, w.bbox.y0 * s + dy, w.bbox.x1 * s + dx, w.bbox.y1 * s + dy), w.conf, w.line_id) for w in words]
    a, b = layout_vector(words), layout_vector(moved)
    assert float(a @ b) > 0.999


def test_layout_separates_different_arrangements_and_handles_empty():
    assert float(layout_vector(_words(1)) @ layout_vector(_words(2))) < 0.95
    assert layout_vector([]).shape == (256,) and not layout_vector([]).any()


# ---- composição do vetor -------------------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(4))
def test_composite_dot_product_is_the_weighted_average_of_block_cosines(seed):
    rng = np.random.default_rng(seed)
    a, b = _vector(rng), _vector(rng)
    weights = {"visual": 0.5, "text": 0.3, "layout": 0.2}
    expected = sum(w * float(a.blocks[n] @ b.blocks[n]) for n, w in weights.items())
    assert float(a.composite(weights) @ b.composite(weights)) == pytest.approx(expected, abs=1e-5)


def test_weights_are_renormalized_over_available_blocks():
    out = normalize_weights({"visual": 0.4, "text": 0.4, "layout": 0.2}, {"text", "layout"})
    assert out == {"text": pytest.approx(2 / 3), "layout": pytest.approx(1 / 3)}
    with pytest.raises(ValueError):
        normalize_weights({"visual": 1.0}, {"text"})


# ---- índice --------------------------------------------------------------------------------------------------

def test_index_ranks_exact_match_first_with_scores_between_0_and_1():
    rng = np.random.default_rng(0)
    index, vectors = NumpyFlatIndex(), {}
    for name in "abcde":
        vectors[name] = _vector(rng)
        index.add(name, vectors[name], {"label": name})
    hits = index.search(vectors["c"], 5, {"visual": 0.4, "text": 0.4, "layout": 0.2})
    assert hits[0].doc_id == "c" and hits[0].score == pytest.approx(1.0, abs=1e-5)
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    assert all(0.0 <= v <= 1.0 for h in hits for v in (h.score, *h.block_scores.values()))


def test_negative_cosine_counts_as_zero():
    index = NumpyFlatIndex()
    index.add("oposto", DocumentVector({"text": np.array([-1, 0], np.float32)}))
    hit = index.search(DocumentVector({"text": np.array([1, 0], np.float32)}), 1, {"text": 1.0})[0]
    assert hit.score == 0.0


def test_reweighting_changes_the_ranking_without_reindexing():
    index = NumpyFlatIndex()
    q = DocumentVector({"text": np.array([1, 0], np.float32), "visual": np.array([1, 0], np.float32)})
    index.add("igual_no_texto", DocumentVector({"text": np.array([1, 0], np.float32), "visual": np.array([0, 1], np.float32)}))
    index.add("igual_no_visual", DocumentVector({"text": np.array([0, 1], np.float32), "visual": np.array([1, 0], np.float32)}))
    assert index.search(q, 1, {"text": 1.0, "visual": 0.0})[0].doc_id == "igual_no_texto"
    assert index.search(q, 1, {"text": 0.0, "visual": 1.0})[0].doc_id == "igual_no_visual"


def test_index_replaces_same_id_rejects_mismatched_blocks_and_uses_shared_blocks_only():
    rng = np.random.default_rng(1)
    index = NumpyFlatIndex()
    index.add("x", _vector(rng))
    index.add("x", _vector(rng))
    assert len(index) == 1
    with pytest.raises(ValueError):
        index.add("y", _vector(rng, visual=False))
    hit = index.search(_vector(rng, visual=False), 1, {"visual": 0.4, "text": 0.4, "layout": 0.2})[0]
    assert set(hit.block_scores) == {"text", "layout"}


def test_index_search_on_empty_index_returns_nothing():
    assert NumpyFlatIndex().search(_vector(np.random.default_rng(0)), 3, {"text": 1.0}) == []


def test_repository_roundtrip_and_empty_load(tmp_path):
    rng = np.random.default_rng(3)
    index = NumpyFlatIndex()
    for name in ("a", "b"):
        index.add(name, _vector(rng), {"label": name})
    repo = DocumentIndexRepository(tmp_path)
    repo.save(index)
    loaded = repo.load()
    assert loaded.ids == ["a", "b"] and loaded.metas == index.metas
    assert np.allclose(loaded.matrices["text"], index.matrices["text"])
    assert len(DocumentIndexRepository(tmp_path / "vazio").load()) == 0
