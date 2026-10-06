"""Busca de imagens por texto: lógica do caso de uso e do índice, com um embedder de mentira (cor no lugar do SigLIP)."""

import numpy as np
import pytest

from app.api.repositories.semantic_index_repository import SemanticIndexRepository
from app.api.services.semantic_index import SemanticIndex
from app.api.settings import Settings
from app.api.use_cases.search_images import SearchImagesUseCase
from app.tests.conftest import ColorEmbedder, make_dress, png


def _index_dresses(use_case):
    items = [(png(make_dress(color, shade)), f"{color}_{shade}.png") for color in ("red", "green", "blue") for shade in (0, 1, 2)]
    use_case.add_many(items)
    return {label for _, label in items}


def test_semantic_index_ranks_by_cosine_and_replaces_same_id():
    index = SemanticIndex()
    index.add("a", np.array([1, 0], np.float32), {"label": "a"})
    index.add("b", np.array([0, 1], np.float32), {"label": "b"})
    assert [h.doc_id for h in index.search(np.array([0.2, 1], np.float32), 2)] == ["b", "a"]
    index.add("a", np.array([0, 1], np.float32), {"label": "a2"})  # mesma imagem: substitui
    assert len(index) == 2 and index.metas[0]["label"] == "a2"
    assert SemanticIndex().search(np.array([1, 0], np.float32), 3) == []


def test_text_query_retrieves_the_images_of_that_color(images_use_case):
    _index_dresses(images_use_case)
    for color in ("red", "green", "blue"):
        hits = images_use_case.search_text(f"{color} dress", k=3).response.hits
        assert {h.label.split("_")[0] for h in hits} == {color}
        assert [h.rank for h in hits] == [1, 2, 3]


def test_ranking_scores_are_descending_and_response_reports_corpus_size(images_use_case):
    _index_dresses(images_use_case)
    response = images_use_case.search_text("green dress", k=9).response
    scores = [h.score for h in response.hits]
    assert scores == sorted(scores, reverse=True) and response.n_indexed == 9 and response.query == "green dress"


def test_outcome_paths_point_to_saved_copies_of_the_images(images_use_case):
    _index_dresses(images_use_case)
    outcome = images_use_case.search_text("red dress", k=2)
    assert len(outcome.paths) == 2 and all(p.exists() and p.stat().st_size > 0 for p in outcome.paths)


def test_indexing_the_same_bytes_again_replaces_it(images_use_case):
    data = png(make_dress("red"))
    first, second = images_use_case.add(data, "a.png"), images_use_case.add(data, "b.png")
    assert not first.replaced and second.replaced and images_use_case.n_indexed == 1


def test_blank_query_is_rejected(images_use_case):
    with pytest.raises(ValueError):
        images_use_case.search_text("   ")


def test_searching_an_empty_index_returns_no_hits(images_use_case):
    assert images_use_case.search_text("red dress").response.hits == []


def test_index_persists_across_instances(tmp_path):
    settings = Settings(semantic_index_dir=tmp_path)
    SearchImagesUseCase(settings, ColorEmbedder(), SemanticIndexRepository(tmp_path)).add(png(make_dress("green")), "g.png")
    reloaded = SearchImagesUseCase(settings, ColorEmbedder(), SemanticIndexRepository(tmp_path))
    assert reloaded.n_indexed == 1 and reloaded.search_text("green dress").response.hits[0].label == "g.png"


def test_an_index_made_by_another_model_is_refused_instead_of_mixed(tmp_path):
    """Vetores de modelos (ou pré-processamentos) diferentes não são comparáveis: erro claro, não ranking errado."""
    settings = Settings(semantic_index_dir=tmp_path)
    SearchImagesUseCase(settings, ColorEmbedder(), SemanticIndexRepository(tmp_path)).add(png(make_dress("red")), "r.png")

    class OtherModel(ColorEmbedder):
        name = "outro-modelo"

    with pytest.raises(ValueError, match="não são compatíveis"):
        SearchImagesUseCase(settings, OtherModel(), SemanticIndexRepository(tmp_path))


def test_a_legacy_index_without_model_name_is_refused(tmp_path):
    settings = Settings(semantic_index_dir=tmp_path)
    repo = SemanticIndexRepository(tmp_path)
    legacy = SemanticIndex()
    legacy.add("x", np.array([1, 0, 0], np.float32), {"label": "x", "path": "x"})
    repo.save(legacy)  # model=None, como os índices criados antes de o modelo ser registrado
    with pytest.raises(ValueError, match="modelo antigo"):
        SearchImagesUseCase(settings, ColorEmbedder(), repo)


def test_an_empty_index_adopts_the_current_model(tmp_path):
    repo = SemanticIndexRepository(tmp_path)
    use_case = SearchImagesUseCase(Settings(semantic_index_dir=tmp_path), ColorEmbedder(), repo)
    use_case.add(png(make_dress("red")), "r.png")
    assert repo.load().model == "fake-color"
