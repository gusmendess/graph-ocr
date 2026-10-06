import numpy as np

from app.api.domain.bbox import BBox
from app.api.domain.entities import Word


class RapidOcrEngine:
    """PP-OCR (modelos ONNX, sem binário externo): devolve palavra, bbox e confiança."""

    name = "rapidocr"

    def __init__(self) -> None:
        from rapidocr import RapidOCR

        self._ocr = RapidOCR()

    def read(self, image: np.ndarray) -> list[Word]:
        result = self._ocr(image, return_word_box=True)
        words: list[Word] = []
        for line_id, line in enumerate(result.word_results or ()):
            for text, score, quad in line:
                if not text.strip():
                    continue
                xs = [p[0] for p in quad]
                ys = [p[1] for p in quad]
                words.append(Word(text.strip(), BBox(min(xs), min(ys), max(xs), max(ys)), float(score), line_id))
        return words
