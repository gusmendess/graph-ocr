import zlib
from typing import Protocol

import numpy as np

from app.api.utils import normalize_text


class Embedder(Protocol):
    """Texto -> vetor L2-normalizado. Troque por um modelo (ex.: Transformer) sem mexer no resto."""

    def embed(self, texts: list[str]) -> np.ndarray: ...


class CharNgramEmbedder:
    """Vetor de n-gramas de caracteres (hashing trick), determinístico e sem download de modelo.

    Escolhido de propósito: embeddings de frase tendem a borrar dígitos ("06/06/2021" ~ "26/06/2021"),
    enquanto n-gramas de caracteres mantêm a diferença visível no cosseno. Uma edição no lugar
    aparece como cosseno ALTO porém menor que 1 ("quase igual"); conteúdo diferente cai muito mais.
    """

    def __init__(self, dim: int = 2048, ngram_range: tuple[int, int] = (2, 4)) -> None:
        self._dim = dim
        self._range = ngram_range

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for row, text in enumerate(texts):
            padded = f" {normalize_text(text)} "
            for n in range(self._range[0], self._range[1] + 1):
                for i in range(len(padded) - n + 1):
                    out[row, zlib.crc32(padded[i : i + n].encode("utf-8")) % self._dim] += 1.0
        return _l2(out)


def _l2(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _l2(a) @ _l2(b).T
