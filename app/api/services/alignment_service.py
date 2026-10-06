import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

from app.api.domain.alignment import Alignment, AlignmentConfig

_Arrays = tuple[np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray]


def _arrays(graph: nx.Graph, k: int) -> _Arrays:
    """Centros (px), alturas (px), assinatura geométrica por nó, chaves de texto e altura mediana."""
    nodes = sorted(graph.nodes)
    words = [graph.nodes[i]["word"] for i in nodes]
    centers = np.array([w.bbox.center for w in words], dtype=float).reshape(-1, 2)
    heights = np.array([w.bbox.height for w in words], dtype=float)
    unit = max(float(np.median(heights)), 1.0) if len(words) else 1.0

    desc = np.zeros((len(nodes), k + 1))
    for row, i in enumerate(nodes):
        # edge weight já é distância / altura mediana do documento: invariante a escala e rotação
        dists = sorted(graph.edges[i, j]["weight"] for j in graph.neighbors(i))[:k]
        if dists:
            dists += [dists[-1]] * (k - len(dists))
            desc[row, :k] = dists
        desc[row, k] = heights[row] / unit
    return centers, heights, desc, [graph.nodes[i]["key"] for i in nodes], np.array(unit)


def _candidates(desc_a, desc_b, keys_a, keys_b, per_node: int) -> list[tuple[int, int]]:
    """Pares (i em A, j em B): vizinhos mais próximos pela assinatura + textos únicos iguais (dica)."""
    tree = cKDTree(desc_b)
    _, idx = tree.query(desc_a, k=min(per_node, len(desc_b)))
    pairs = {(i, int(j)) for i, row in enumerate(np.atleast_2d(idx)) for j in np.atleast_1d(row)}
    only_a = {k: i for i, k in enumerate(keys_a) if keys_a.count(k) == 1}
    only_b = {k: j for j, k in enumerate(keys_b) if keys_b.count(k) == 1}
    pairs |= {(only_a[k], only_b[k]) for k in only_a.keys() & only_b.keys()}
    return sorted(pairs)


def _similarity_from_pairs(a1, a2, b1, b2) -> np.ndarray | None:
    """Similaridade (escala, rotação, translação) que leva b1,b2 em a1,a2 (números complexos)."""
    zb = complex(*(b2 - b1))
    if abs(zb) < 1e-9:
        return None
    w = complex(*(a2 - a1)) / zb
    t = complex(*a1) - w * complex(*b1)
    return np.array([[w.real, -w.imag, t.real], [w.imag, w.real, t.imag]])


def _fit(pb: np.ndarray, pa: np.ndarray, model: str) -> np.ndarray:
    if model == "affine" and len(pb) >= 3:
        x = np.hstack([pb, np.ones((len(pb), 1))])
        return np.linalg.lstsq(x, pa, rcond=None)[0].T
    mb, ma = pb.mean(axis=0), pa.mean(axis=0)
    zb = (pb - mb) @ [1, 1j]
    za = (pa - ma) @ [1, 1j]
    w = (np.conj(zb) * za).sum() / max((np.abs(zb) ** 2).sum(), 1e-12)
    t = complex(*ma) - w * complex(*mb)
    return np.array([[w.real, -w.imag, t.real], [w.imag, w.real, t.imag]])


def _scale(m: np.ndarray) -> float:
    return float(np.sqrt(abs(np.linalg.det(m[:, :2]))))


class _Scorer:
    """Mede o quanto o grafo de B, transformado, sobrepõe o grafo de A."""

    def __init__(self, a: _Arrays, b: _Arrays, tol_units: float, text_weight: float) -> None:
        self.ca, self.ha, _, self.ka, self.unit_a = a
        self.cb, self.hb, _, self.kb, _ = b
        self.tree = cKDTree(self.ca)
        self.tol = tol_units * float(self.unit_a)
        self.text_weight = text_weight
        self.ka_arr, self.kb_arr = np.array(self.ka), np.array(self.kb)

    def inliers(self, m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        moved = self.cb @ m[:, :2].T + m[:, 2]
        dist, idx = self.tree.query(moved)
        size_ok = np.abs(self.hb * _scale(m) - self.ha[idx]) <= 0.5 * self.ha[idx]
        return (dist <= self.tol) & size_ok, idx

    def score(self, m: np.ndarray) -> float:
        ok, idx = self.inliers(m)
        same_text = ok & (self.ka_arr[idx] == self.kb_arr)
        return float(ok.sum() + self.text_weight * same_text.sum())


def estimate_alignment(graph_a: nx.Graph, graph_b: nx.Graph, cfg: AlignmentConfig, k: int = 4) -> Alignment:
    """Sobrepõe o grafo de B ao de A e devolve a transformação B -> A.

    1. Cada nó ganha uma assinatura geométrica (distâncias aos vizinhos / altura de palavra, altura).
       Pares candidatos A↔B saem dessa assinatura, sem exigir texto igual.
    2. RANSAC: 2 pares candidatos definem uma similaridade; o placar é quantos nós de B caem sobre
       nós de A com tamanho compatível (bônus opcional por texto igual).
    3. Refinamento por mínimos quadrados sobre os inliers (similaridade ou afim, conforme `cfg.model`).

    Palavras alteradas, inseridas ou removidas viram outliers e não distorcem o ajuste.
    """
    if graph_a.number_of_nodes() < 2 or graph_b.number_of_nodes() < 2:
        return Alignment.identity()

    a, b = _arrays(graph_a, k), _arrays(graph_b, k)
    cands = _candidates(a[2], b[2], a[3], b[3], cfg.candidates_per_node)
    scorer = _Scorer(a, b, cfg.inlier_tolerance, cfg.text_weight)
    min_sep = cfg.min_separation * float(b[4])
    rng = np.random.default_rng(cfg.seed)

    best, best_score = None, 0.0
    if len(cands) >= 2:
        for _ in range(cfg.iterations):
            (i1, j1), (i2, j2) = (cands[x] for x in rng.choice(len(cands), size=2, replace=False))
            if i1 == i2 or j1 == j2 or np.linalg.norm(b[0][j2] - b[0][j1]) < min_sep:
                continue
            m = _similarity_from_pairs(a[0][i1], a[0][i2], b[0][j1], b[0][j2])
            if m is None or not 1 / cfg.max_scale <= _scale(m) <= cfg.max_scale:
                continue
            score = scorer.score(m)
            if score > best_score:
                best, best_score = m, score
    if best is None:
        return Alignment.identity(len(cands))

    m = best
    models = ["similarity"] + ([cfg.model] if cfg.model != "similarity" else [])  # afim parte da similaridade
    for model in models:
        for _ in range(3):  # refina: reatribui inliers e reajusta
            ok, idx = scorer.inliers(m)
            if ok.sum() < 2:
                break
            candidate = _fit(scorer.cb[ok], scorer.ca[idx[ok]], model)
            if 1 / cfg.max_scale <= _scale(candidate) <= cfg.max_scale:
                m = candidate
    ok, _ = scorer.inliers(m)
    n_in = int(ok.sum())
    if n_in < cfg.min_inliers:
        return Alignment.identity(len(cands))
    return Alignment(m, cfg.model, n_in, len(cands), True)
