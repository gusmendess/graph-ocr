"""Alinhamento de grafos: layouts sintéticos gerados aleatoriamente, transformações sorteadas por teste."""

import math

import numpy as np
import pytest

from app.api.domain.alignment import Alignment, AlignmentConfig
from app.api.domain.bbox import BBox
from app.api.domain.entities import Word
from app.api.domain.enums import WordStatus
from app.api.services import alignment_service, forensics_service, graph_service, matching_service


def _layout(rng: np.random.Generator, n_lines: int = 14, label: str = "w") -> list[Word]:
    """Documento sintético: linhas de texto com larguras, espaçamentos e alturas variados."""
    words, y, idx = [], float(rng.uniform(20, 60)), 0
    for _ in range(n_lines):
        x, h = float(rng.uniform(15, 50)), float(rng.uniform(18, 30))
        for _ in range(int(rng.integers(1, 5))):
            w = float(rng.uniform(25, 160))
            words.append(Word(f"{label}{idx}", BBox(x, y, x + w, y + h), 0.99))
            x += w + float(rng.uniform(8, 20))
            idx += 1
        y += h + float(rng.uniform(14, 40))
    return words


def _transform(words: list[Word], scale: float, deg: float, tx: float, ty: float) -> tuple[list[Word], np.ndarray]:
    """Aplica uma similaridade aos CENTROS das palavras (tamanho escala junto). Devolve também a matriz."""
    c, s = scale * math.cos(math.radians(deg)), scale * math.sin(math.radians(deg))
    m = np.array([[c, -s, tx], [s, c, ty]])
    out = []
    for w in words:
        cx, cy = m[:, :2] @ np.array(w.bbox.center) + m[:, 2]
        hw, hh = w.bbox.width * scale / 2, w.bbox.height * scale / 2
        out.append(Word(w.text, BBox(cx - hw, cy - hh, cx + hw, cy + hh), w.conf))
    return out, m


def _align(a: list[Word], b: list[Word], **cfg) -> Alignment:
    ga, gb = graph_service.build_word_graph(a), graph_service.build_word_graph(b)
    return alignment_service.estimate_alignment(ga, gb, AlignmentConfig(**cfg))


def _inverse(m: np.ndarray) -> np.ndarray:
    full = np.vstack([m, [0, 0, 1]])
    return np.linalg.inv(full)[:2]


@pytest.mark.parametrize("seed", range(6))
def test_recovers_random_similarity_without_text(seed):
    """B tem textos DIFERENTES de A: o alinhamento precisa vir só da geometria dos grafos."""
    rng = np.random.default_rng(seed)
    a = _layout(rng, label="a")
    scale, deg = float(rng.uniform(0.85, 1.2)), float(rng.uniform(-6, 6))
    tx, ty = (float(v) for v in rng.uniform(-80, 80, 2))
    b_moved, m_ab = _transform(a, scale, deg, tx, ty)  # A -> B
    b = [Word(f"x{i}", w.bbox, w.conf) for i, w in enumerate(b_moved)]

    al = _align(a, b, text_weight=0.0)

    assert al.converged and al.inliers >= len(a) * 0.9
    expected = _inverse(m_ab)  # B -> A
    pts = np.array([w.bbox.center for w in b])
    err = np.linalg.norm(al.apply_points(pts) - np.array([w.bbox.center for w in a]), axis=1)
    assert err.max() < 2.0  # px


def test_outliers_do_not_break_alignment():
    rng = np.random.default_rng(42)
    a = _layout(rng)
    b, _ = _transform(a, 1.05, 2.0, 40.0, -25.0)
    b = b[:-4] + [Word("novo", BBox(300, 900, 380, 925), 0.9)]  # remove 4 palavras e insere 1
    b[3] = Word("EDIT", b[3].bbox, 0.4)  # texto alterado
    al = _align(a, b)
    assert al.converged
    assert abs(al.scale - 1 / 1.05) < 0.02 and abs(al.rotation_deg + 2.0) < 0.3


def test_identical_layout_gives_identity():
    a = _layout(np.random.default_rng(1))
    al = _align(a, a)
    assert al.converged
    assert abs(al.scale - 1) < 1e-3 and abs(al.rotation_deg) < 1e-2 and np.allclose(al.translation, 0, atol=1e-2)


def test_too_few_nodes_falls_back_to_identity():
    a = _layout(np.random.default_rng(3), n_lines=1)[:1]
    al = _align(a, a)
    assert not al.converged and np.allclose(al.matrix, Alignment.identity().matrix)


def test_affine_model_handles_anisotropic_scale():
    rng = np.random.default_rng(7)
    a = _layout(rng, n_lines=18)
    b = []
    for w in a:  # estica só o eixo y em 8%
        b.append(Word(w.text, BBox(w.bbox.x0, w.bbox.y0 * 1.08, w.bbox.x1, w.bbox.y1 * 1.08), w.conf))
    al = _align(a, b, model="affine")
    assert al.converged
    pa = np.array([w.bbox.center for w in a])
    err = np.linalg.norm(al.apply_points(np.array([w.bbox.center for w in b])) - pa, axis=1)
    assert np.median(err) < 1.5


@pytest.mark.parametrize("seed", range(4))
def test_edit_detected_after_random_transform(seed):
    """Fim a fim: documento deslocado/rotacionado + uma edição -> exatamente 1 MODIFIED, resto MATCHED."""
    rng = np.random.default_rng(100 + seed)
    a = _layout(rng, label="t")
    b, _ = _transform(
        a, float(rng.uniform(0.9, 1.15)), float(rng.uniform(-4, 4)), *(float(v) for v in rng.uniform(-70, 70, 2))
    )
    target = int(rng.integers(len(b)))
    b[target] = Word("EDITADO", b[target].bbox, b[target].conf)

    ga, gb = graph_service.build_word_graph(a), graph_service.build_word_graph(b)
    al = alignment_service.estimate_alignment(ga, gb, AlignmentConfig())
    matches = matching_service.match_words(a, b, ga, gb, alignment=al)
    forensics_service.annotate_matches(matches, gb, al)

    statuses = [m.status for m in matches]
    assert statuses.count(WordStatus.MODIFIED) == 1
    assert statuses.count(WordStatus.MATCHED) >= len(a) - 3
    mod = next(m for m in matches if m.status == WordStatus.MODIFIED)
    assert abs(mod.deltas.baseline_px) < 3 and abs(mod.deltas.height_pct) < 8  # deslocamento global não vira "sinal"


def test_without_alignment_a_shifted_doc_is_not_matched():
    """Controle: sem o alinhamento o mesmo cenário falha, então o teste acima exercita o alinhamento."""
    rng = np.random.default_rng(5)
    a = _layout(rng, label="t")
    b, _ = _transform(a, 1.0, 0.0, 60.0, 50.0)
    ga, gb = graph_service.build_word_graph(a), graph_service.build_word_graph(b)
    raw = matching_service.match_words(a, b, ga, gb)
    assert sum(m.status == WordStatus.MATCHED for m in raw) < len(a) * 0.5
