from __future__ import annotations

from dataclasses import dataclass

from app.api.domain.bbox import BBox


@dataclass(frozen=True)
class Word:
    """Uma palavra lida pelo OCR: texto + bounding box (pixels) + confiança em [0, 1]."""

    text: str
    bbox: BBox
    conf: float
    line_id: int | None = None
