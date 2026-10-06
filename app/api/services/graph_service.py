import numpy as np
import networkx as nx

from app.api.domain.entities import Word
from app.api.models.schemas import Entity, SharedEntity
from app.api.utils import normalize_text


def build_word_graph(words: list[Word], k: int = 4) -> nx.Graph:
    """Grafo k-NN espacial: nó = palavra do OCR, aresta = vizinhança geométrica.

    A distância é medida em "alturas de palavra" (mediana), então não depende da resolução.
    O contexto (quem está ao lado) é o que desambigua palavras repetidas como "R$" ou "do".
    """
    graph = nx.Graph()
    for i, word in enumerate(words):
        graph.add_node(i, word=word, key=normalize_text(word.text))
    if len(words) < 2:
        return graph

    centers = np.array([w.bbox.center for w in words])
    unit = max(float(np.median([w.bbox.height for w in words])), 1.0)
    dist = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1) / unit
    np.fill_diagonal(dist, np.inf)
    for i in range(len(words)):
        for j in np.argsort(dist[i])[: min(k, len(words) - 1)]:
            graph.add_edge(i, int(j), weight=float(dist[i, j]))
    return graph


def neighbor_keys(graph: nx.Graph, node: int) -> set[str]:
    return {graph.nodes[n]["key"] for n in graph.neighbors(node)}


def jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def same_line_neighbors(graph: nx.Graph, node: int) -> list[Word]:
    word: Word = graph.nodes[node]["word"]
    cy = word.bbox.center[1]
    out = []
    for n in graph.neighbors(node):
        other: Word = graph.nodes[n]["word"]
        if abs(other.bbox.center[1] - cy) < 0.6 * max(word.bbox.height, other.bbox.height):
            out.append(other)
    return out


def build_case_graph(entities_by_doc: dict[str, list[Entity]]) -> nx.Graph:
    """Grafo bipartido documento <-> entidade (chave Pix, CNPJ, ID de transação...)."""
    graph = nx.Graph()
    for doc, entities in entities_by_doc.items():
        graph.add_node(("doc", doc), kind="doc")
        for entity in entities:
            node = (entity.kind, entity.value)
            graph.add_node(node, kind=entity.kind)
            graph.add_edge(("doc", doc), node)
    return graph


def shared_entities(graph: nx.Graph) -> list[SharedEntity]:
    shared = []
    for node, attrs in graph.nodes(data=True):
        if attrs["kind"] == "doc" or graph.degree(node) < 2:
            continue
        docs = sorted(d[1] for d in graph.neighbors(node))
        shared.append(SharedEntity(kind=node[0], value=node[1], documents=docs))
    return shared
