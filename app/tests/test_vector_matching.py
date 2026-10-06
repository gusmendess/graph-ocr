"""Pareamento por vetor de texto + cosseno (+ contexto do grafo): layouts sintéticos aleatórios, sem OCR."""

import random
import string

import numpy as np
import pytest

from app.api.domain.bbox import BBox
from app.api.domain.entities import Word
from app.api.domain.enums import WordStatus
from app.api.services import vector_matching_service as vm
from app.api.services.embedding_service import CharNgramEmbedder, cosine_matrix

EMB = CharNgramEmbedder()


def _rand(rng: random.Random, n: int, alphabet: str) -> str:
    return "".join(rng.choice(alphabet) for _ in range(n))


def _receipt(seed: int, n_fields: int = 14) -> list[Word]:
    """Comprovante sintético: pares rótulo/valor empilhados; cada linha = um line_id."""
    rng = random.Random(seed)
    words, y, line = [], 30.0, 0
    for _ in range(n_fields):
        for kind in ("label", "value"):
            text = (
                " ".join(_rand(rng, rng.randint(3, 9), string.ascii_lowercase) for _ in range(rng.randint(1, 3))).capitalize()
                if kind == "label"
                else rng.choice([_rand(rng, rng.randint(8, 26), string.digits), _rand(rng, 4, string.digits) + "/" + _rand(rng, 6, string.digits)])
            )
            x = 40.0
            for tok in text.split(" "):
                w = 9.0 * len(tok)
                words.append(Word(tok, BBox(x, y, x + w, y + 22), 0.99, line))
                x += w + 10
            line += 1
            y += 34
        y += 26
    return words


def _replace(words: list[Word], line_id: int, new_text: str) -> list[Word]:
    """Troca o conteúdo de uma linha inteira (um único token), mantendo a posição."""
    mine = [w for w in words if w.line_id == line_id]
    box = BBox(mine[0].bbox.x0, mine[0].bbox.y0, mine[-1].bbox.x1, mine[0].bbox.y1)
    return [w for w in words if w.line_id != line_id] + [Word(new_text, box, 0.99, line_id)]


def _statuses(matches):
    return [m.status for m in matches]


def test_embedder_keeps_digit_differences_visible():
    near = cosine_matrix(EMB.embed(["06/06/2021"]), EMB.embed(["26/06/2021"]))[0, 0]
    far = cosine_matrix(EMB.embed(["06/06/2021"]), EMB.embed(["Chave"]))[0, 0]
    assert far < near < 1.0


@pytest.mark.parametrize("seed", range(5))
def test_identical_documents_are_fully_matched(seed):
    doc = _receipt(seed)
    assert set(_statuses(vm.match_by_vectors(doc, doc, EMB))) == {WordStatus.MATCHED}


@pytest.mark.parametrize("seed", range(5))
def test_small_edit_is_modified_with_high_but_imperfect_cosine(seed):
    a = _receipt(seed)
    value_line = max(w.line_id for w in a) - 1  # uma linha de valor
    old = " ".join(w.text for w in a if w.line_id == value_line)
    new = old[:-1] + ("0" if old[-1] != "0" else "1")
    m = vm.match_by_vectors(a, _replace(a, value_line, new), EMB)
    mod = [x for x in m if x.status == WordStatus.MODIFIED]
    assert len(mod) == 1 and mod[0].a.text != mod[0].b.text
    assert 0.3 <= mod[0].cosine < 0.995


@pytest.mark.parametrize("seed", range(5))
def test_fully_replaced_values_are_modified_even_when_adjacent(seed):
    """Fraudador troca vários valores por inteiro (sem semelhança alguma): continuam sendo o MESMO campo."""
    a = _receipt(seed)
    rng = random.Random(seed + 99)
    targets = [ln for ln in sorted({w.line_id for w in a}) if ln % 2 == 1][2:5]  # 3 valores vizinhos
    b = a
    for ln in targets:
        b = _replace(b, ln, _rand(rng, 20, string.ascii_uppercase))
    statuses = _statuses(vm.match_by_vectors(a, b, EMB))
    assert statuses.count(WordStatus.MODIFIED) >= len(targets)
    assert WordStatus.REMOVED not in statuses and WordStatus.ADDED not in statuses


@pytest.mark.parametrize("seed", range(3))
def test_global_shift_and_scale_do_not_matter(seed):
    a = _receipt(seed)
    b = [Word(w.text, BBox(w.bbox.x0 * 1.7 + 300, w.bbox.y0 * 1.7 - 80, w.bbox.x1 * 1.7 + 300, w.bbox.y1 * 1.7 - 80), w.conf, w.line_id) for w in a]
    assert set(_statuses(vm.match_by_vectors(a, b, EMB))) == {WordStatus.MATCHED}


def test_removed_and_added_lines():
    a = _receipt(1)
    gone = max(w.line_id for w in a)
    b = [w for w in a if w.line_id != gone]
    assert WordStatus.REMOVED in _statuses(vm.match_by_vectors(a, b, EMB))
    assert WordStatus.ADDED in _statuses(vm.match_by_vectors(b, a, EMB))


def test_unrelated_documents_share_almost_nothing():
    m = vm.match_by_vectors(_receipt(1), _receipt(2), EMB)
    matched = sum(s == WordStatus.MATCHED for s in _statuses(m))
    assert matched / len(m) < 0.2


def test_slot_rule_can_be_disabled():
    a = _receipt(3)
    b = _replace(a, 5, "ZZZZZZZZZZZZZZZZZZZZ")
    off = _statuses(vm.match_by_vectors(a, b, EMB, slot_overlap=1.1))  # Jaccard nunca passa de 1
    assert WordStatus.REMOVED in off and WordStatus.ADDED in off
