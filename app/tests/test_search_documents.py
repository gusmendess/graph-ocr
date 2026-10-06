"""Cascata sha256 -> phash -> vetor, com OCR de mentira e sem o bloco visual (sem modelos nem rede)."""

import pytest

from app.api.services import hashing_service as hs
from app.tests.conftest import make_document, png


def _phash_sim(a, b) -> float:
    return hs.phash_similarity(hs.phash(a), hs.phash(b))


def test_identical_bytes_are_decided_by_sha256_with_score_one(documents_use_case):
    use_case = documents_use_case()
    data = png(make_document(1))
    use_case.add(data, "um.png")
    response = use_case.search(data, k=5).response
    assert response.exact_match
    assert response.hits[0].decided_by == "sha256" and response.hits[0].score == 1.0 and response.hits[0].is_query
    assert response.hits[0].phash is None


def test_visually_equal_but_different_bytes_go_to_the_vector_stage(documents_use_case):
    use_case = documents_use_case()
    use_case.add(png(make_document(1)), "um.png")
    response = use_case.search(png(make_document(2)), k=5).response  # 1 pixel de diferença
    hit = response.hits[0]
    assert not response.exact_match and hit.decided_by == "vector"
    assert response.vector_computed and set(hit.block_scores) == {"text", "layout"}
    assert 0.0 <= hit.score <= 1.0 and hit.phash == pytest.approx(1.0)


def test_phash_bars_a_very_different_page_without_running_ocr_or_vector(documents_use_case, ocr_engine):
    similar, different = make_document(1), make_document(2, inverted=True)
    threshold = (_phash_sim(similar, make_document(3)) + _phash_sim(similar, different)) / 2
    assert _phash_sim(similar, different) < threshold < _phash_sim(similar, make_document(3))  # a premissa do teste

    use_case = documents_use_case(phash_threshold=threshold)
    use_case.add(png(similar), "claro.png")
    calls_after_indexing = ocr_engine.calls
    response = use_case.search(png(different), k=5).response

    hit = response.hits[0]
    assert hit.decided_by == "phash" and hit.score == pytest.approx(hit.phash) and hit.block_scores == {}
    assert not response.vector_computed and response.blocks_used == []
    assert ocr_engine.calls == calls_after_indexing  # o OCR não rodou para a consulta


def test_vector_runs_only_for_pairs_that_pass_the_phash_gate(documents_use_case):
    similar, different = make_document(1), make_document(2, inverted=True)
    threshold = (_phash_sim(similar, make_document(3)) + _phash_sim(similar, different)) / 2
    use_case = documents_use_case(phash_threshold=threshold)
    use_case.add(png(similar), "claro.png")
    use_case.add(png(different), "escuro.png")
    response = use_case.search(png(make_document(3)), k=5).response
    by_label = {h.label: h for h in response.hits}
    assert by_label["claro.png"].decided_by == "vector" and by_label["escuro.png"].decided_by == "phash"
    # quem passou pelos portões vem antes de quem foi barrado, mesmo que o score do phash fosse maior
    assert [h.decided_by for h in response.hits] == ["vector", "phash"]


def test_stage_counts_cover_every_indexed_document_not_just_the_top_k(documents_use_case):
    use_case = documents_use_case()
    data = png(make_document(1))
    for tag in range(1, 6):
        use_case.add(png(make_document(tag)), f"doc{tag}.png")
    response = use_case.search(data, k=2).response
    assert len(response.hits) == 2 and response.n_indexed == 5
    assert sum(response.stage_counts.values()) == 5 and response.stage_counts == {"sha256": 1, "vector": 4}


def test_phash_threshold_zero_sends_every_pair_to_the_vector(documents_use_case):
    use_case = documents_use_case(phash_threshold=0.0)
    use_case.add(png(make_document(1)), "claro.png")
    use_case.add(png(make_document(2, inverted=True)), "escuro.png")
    response = use_case.search(png(make_document(3)), k=5).response
    assert {h.decided_by for h in response.hits} == {"vector"}


def test_indexing_the_same_bytes_twice_replaces_instead_of_duplicating(documents_use_case):
    use_case = documents_use_case()
    data = png(make_document(1))
    first = use_case.add(data, "a.png")
    second = use_case.add(data, "b.png")
    assert not first.replaced and second.replaced and use_case.n_indexed == 1
    assert use_case.search(data).response.hits[0].label == "b.png"  # vale o rótulo mais recente


def test_index_persists_and_reloads_from_disk(documents_use_case):
    use_case = documents_use_case()
    use_case.add(png(make_document(1)), "a.png")
    reloaded = documents_use_case()  # nova instância sobre a mesma pasta
    assert reloaded.n_indexed == 1
    assert reloaded.search(png(make_document(1))).response.exact_match


def test_entries_without_a_stored_phash_are_handled(documents_use_case):
    use_case = documents_use_case()
    use_case.add(png(make_document(1)), "a.png")
    use_case.index.metas[0].pop("phash")  # índice antigo, de antes de o phash ser guardado
    response = use_case.search(png(make_document(2))).response
    assert response.hits[0].phash == pytest.approx(1.0)
    assert "phash" in use_case.index.metas[0]  # recalculado a partir da cópia guardada


def test_custom_weights_are_applied_and_reported(documents_use_case):
    use_case = documents_use_case()
    use_case.add(png(make_document(1)), "a.png")
    response = use_case.search(png(make_document(2)), weights={"visual": 0.0, "text": 1.0, "layout": 0.0}).response
    assert response.weights_used == {"text": pytest.approx(1.0), "layout": pytest.approx(0.0)}


def test_search_on_an_empty_index_returns_no_hits(documents_use_case):
    response = documents_use_case().search(png(make_document(1))).response
    assert response.hits == [] and not response.exact_match and not response.vector_computed
