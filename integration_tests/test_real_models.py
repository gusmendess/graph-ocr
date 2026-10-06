import pytest

# data/samples/dresses: example1..3 são vestidos verdes e example4..6, vermelhos
GREEN = {"example1.png", "example2.png", "example3.png"}
RED = {"example4.png", "example5.png", "example6.png"}


def test_a_document_finds_itself_by_sha256_and_everything_else_scores_between_0_and_1(documents_use_case, document_files):
    query = document_files[0]
    response = documents_use_case.search(query.read_bytes(), k=len(document_files)).response

    assert response.exact_match and response.n_indexed == len(document_files)
    top = response.hits[0]
    assert top.label == query.name and top.decided_by == "sha256" and top.score == 1.0
    assert all(0.0 <= h.score <= 1.0 for h in response.hits)
    assert sum(response.stage_counts.values()) == len(document_files)


def test_the_vector_stage_uses_all_three_blocks(documents_use_case, document_files):
    query = document_files[-1].read_bytes() + b"\x00"  # bytes diferentes (o sha256 não decide), imagem igual
    response = documents_use_case.search(query, k=3).response

    assert response.vector_computed and set(response.blocks_used) == {"visual", "text", "layout"}
    best = response.hits[0]
    assert best.decided_by == "vector" and best.score > 0.95  # a mesma imagem, com 1 byte a mais
    assert best.label == document_files[-1].name


@pytest.mark.parametrize(("query", "expected"), [("green dress", GREEN), ("red dress", RED)])
def test_the_natural_language_query_ranks_the_right_color_first(images_use_case, dress_files, query, expected):
    if not expected <= {f.name for f in dress_files}:
        pytest.skip("os exemplos em data/samples/dresses não são os esperados por este teste")
    hits = images_use_case.search_text(query, k=len(dress_files)).response.hits
    assert {h.label for h in hits[: len(expected)]} == expected  # o ranking importa, não o valor do cosseno
    assert min(h.score for h in hits[: len(expected)]) > max(h.score for h in hits[len(expected) :])
