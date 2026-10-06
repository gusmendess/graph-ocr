import shutil
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

from app.api.domain.bbox import BBox
from app.api.domain.entities import Word

_WINDOWS_PATHS = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
)


def find_tesseract(explicit: str | None = None) -> str | None:
    if explicit and Path(explicit).exists():
        return explicit
    found = shutil.which("tesseract")
    if found:
        return found
    return next((p for p in _WINDOWS_PATHS if Path(p).exists()), None)


class TesseractEngine:
    name = "tesseract"

    def __init__(self, cmd: str | None = None, lang: str = "por+eng") -> None:
        resolved = find_tesseract(cmd)
        if resolved is None:
            raise RuntimeError("Binário do Tesseract não encontrado (defina GRAPH_OCR_TESSERACT_CMD)")
        pytesseract.pytesseract.tesseract_cmd = resolved
        available = set(pytesseract.get_languages())
        wanted = [code for code in lang.split("+") if code in available]
        self._lang = "+".join(wanted) or "eng"

    def read(self, image: np.ndarray) -> list[Word]:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        data = pytesseract.image_to_data(rgb, lang=self._lang, output_type=Output.DICT)
        words: list[Word] = []
        for i, text in enumerate(data["text"]):
            conf = float(data["conf"][i])
            if not text.strip() or conf < 0:
                continue
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            line_id = data["block_num"][i] * 10_000 + data["par_num"][i] * 100 + data["line_num"][i]
            words.append(Word(text.strip(), BBox(x, y, x + w, y + h), conf / 100.0, line_id))
        return words
