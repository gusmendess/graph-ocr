import json
from pathlib import Path

from app.api.domain.bbox import BBox
from app.api.domain.entities import Word


class OcrCacheRepository:
    """Cache em disco do OCR, chaveado por sha256 da imagem + motor (demo ao vivo sem esperar o OCR)."""

    def __init__(self, directory: Path) -> None:
        self._dir = directory

    def _path(self, sha: str, engine: str) -> Path:
        return self._dir / f"{sha}.{engine}.json"

    def get(self, sha: str, engine: str) -> list[Word] | None:
        path = self._path(sha, engine)
        if not path.exists():
            return None
        rows = json.loads(path.read_text(encoding="utf-8"))
        return [Word(r["text"], BBox(*r["bbox"]), r["conf"], r["line_id"]) for r in rows]

    def put(self, sha: str, engine: str, words: list[Word]) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        rows = [
            {"text": w.text, "bbox": [w.bbox.x0, w.bbox.y0, w.bbox.x1, w.bbox.y1], "conf": w.conf, "line_id": w.line_id}
            for w in words
        ]
        self._path(sha, engine).write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
