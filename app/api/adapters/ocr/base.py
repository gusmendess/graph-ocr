from typing import Protocol

import numpy as np

from app.api.domain.entities import Word


class OcrEngine(Protocol):
    """Contrato único: imagem BGR -> palavras com bbox (px) e confiança [0, 1]."""

    name: str

    def read(self, image: np.ndarray) -> list[Word]: ...
