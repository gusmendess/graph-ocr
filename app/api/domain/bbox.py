from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BBox:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2

    def iou(self, other: BBox) -> float:
        iw = min(self.x1, other.x1) - max(self.x0, other.x0)
        ih = min(self.y1, other.y1) - max(self.y0, other.y0)
        if iw <= 0 or ih <= 0:
            return 0.0
        inter = iw * ih
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def normalized(self, width: int, height: int) -> BBox:
        """Coordenadas em [0, 1]: permite comparar documentos de resoluções diferentes."""
        return BBox(self.x0 / width, self.y0 / height, self.x1 / width, self.y1 / height)
