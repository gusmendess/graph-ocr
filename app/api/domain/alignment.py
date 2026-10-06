from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from app.api.domain.bbox import BBox


@dataclass(frozen=True)
class AlignmentConfig:
    """Parâmetros do alinhamento de grafos (todos vêm de Settings; nada é específico de um documento)."""

    model: str = "similarity"  # similarity (escala+rotação+translação) | affine
    iterations: int = 2000  # hipóteses RANSAC
    inlier_tolerance: float = 0.4  # distância máxima de centros, em alturas medianas de palavra
    min_inliers: int = 4  # abaixo disso o alinhamento é considerado não convergido
    candidates_per_node: int = 5  # pares candidatos por nó pela assinatura geométrica
    text_weight: float = 0.5  # bônus por texto igual no placar (0 = alinhamento puramente geométrico)
    max_scale: float = 4.0  # escala aceita entre 1/max_scale e max_scale
    min_separation: float = 3.0  # distância mínima entre os 2 pares da amostra, em alturas de palavra
    seed: int = 0


@dataclass(frozen=True)
class Alignment:
    """Transformação 2x3 que leva coordenadas (px) de B para o referencial de A."""

    matrix: np.ndarray
    model: str = "identity"
    inliers: int = 0
    candidates: int = 0
    converged: bool = False

    @classmethod
    def identity(cls, candidates: int = 0) -> Alignment:
        return cls(np.array([[1.0, 0, 0], [0, 1.0, 0]]), candidates=candidates)

    @property
    def scale(self) -> float:
        return math.sqrt(abs(float(np.linalg.det(self.matrix[:, :2]))))

    @property
    def rotation_deg(self) -> float:
        return math.degrees(math.atan2(self.matrix[1, 0], self.matrix[0, 0]))

    @property
    def translation(self) -> tuple[float, float]:
        return float(self.matrix[0, 2]), float(self.matrix[1, 2])

    def apply_points(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(points, dtype=float) @ self.matrix[:, :2].T + self.matrix[:, 2]

    def apply_bbox(self, box: BBox) -> BBox:
        """Move o centro pela transformação e escala largura/altura pelos fatores de cada eixo.

        Não usa a caixa envolvente dos cantos rotacionados: ela infla a altura de palavras largas
        quando há rotação e geraria deltas de altura/baseline falsos.
        """
        cx, cy = self.apply_points(np.array([box.center]))[0]
        sx = math.hypot(self.matrix[0, 0], self.matrix[1, 0])
        sy = math.hypot(self.matrix[0, 1], self.matrix[1, 1])
        hw, hh = box.width * sx / 2, box.height * sy / 2
        return BBox(cx - hw, cy - hh, cx + hw, cy + hh)
