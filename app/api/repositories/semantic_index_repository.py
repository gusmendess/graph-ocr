import json
from pathlib import Path

import numpy as np

from app.api.services.semantic_index import SemanticIndex


class SemanticIndexRepository:
    """Persistência do índice de imagens: vetores (npz), metadados (json) e uma cópia de cada imagem."""

    def __init__(self, directory: Path) -> None:
        self._dir = directory

    def load(self) -> SemanticIndex:
        index = SemanticIndex()
        vectors, meta = self._dir / "vectors.npz", self._dir / "meta.json"
        if vectors.exists() and meta.exists():
            with np.load(vectors) as data:
                index.matrix = data["matrix"]
            stored = json.loads(meta.read_text(encoding="utf-8"))
            index.ids, index.metas = stored["ids"], stored["metas"]
            index.model = stored.get("model")  # índices antigos não têm: foram feitos com o modelo quantizado
        return index

    def save(self, index: SemanticIndex) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        np.savez(self._dir / "vectors.npz", matrix=index.matrix)
        (self._dir / "meta.json").write_text(
            json.dumps({"model": index.model, "ids": index.ids, "metas": index.metas}, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )

    def save_image(self, sha: str, data: bytes, suffix: str) -> Path:
        images = self._dir / "images"
        images.mkdir(parents=True, exist_ok=True)
        path = images / f"{sha}{suffix}"
        path.write_bytes(data)
        return path
