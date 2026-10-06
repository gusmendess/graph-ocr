import numpy as np

from app.api.adapters.ocr.base import OcrEngine
from app.api.domain.entities import Word
from app.api.repositories.ocr_cache import OcrCacheRepository


class OcrService:
    def __init__(self, engine: OcrEngine, cache: OcrCacheRepository | None = None) -> None:
        self._engine = engine
        self._cache = cache

    @property
    def engine_name(self) -> str:
        return self._engine.name

    def read(self, sha: str, image: np.ndarray) -> list[Word]:
        if self._cache is not None:
            cached = self._cache.get(sha, self._engine.name)
            if cached is not None:
                return cached
        words = self._engine.read(image)
        if self._cache is not None:
            self._cache.put(sha, self._engine.name, words)
        return words
