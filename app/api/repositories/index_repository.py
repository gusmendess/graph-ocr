import json
from pathlib import Path

import numpy as np

from app.api.services.vector_index import NumpyFlatIndex


class DocumentIndexRepository:
    """Persistência do índice em disco: vetores (npz), metadados (json) e uma cópia de cada documento.

    A cópia do original fica junto do índice, então a busca consegue mandar os candidatos para o
    OCR + grafo mesmo que o arquivo de origem tenha mudado de lugar.
    """

    def __init__(self, directory: Path) -> None:
        self._dir = directory

    def load(self) -> NumpyFlatIndex:
        index = NumpyFlatIndex()
        vectors, meta = self._dir / "vectors.npz", self._dir / "meta.json"
        if vectors.exists() and meta.exists():
            with np.load(vectors) as data:
                index.matrices = {name: data[name] for name in data.files}
            stored = json.loads(meta.read_text(encoding="utf-8"))
            index.ids, index.metas = stored["ids"], stored["metas"]
        return index

    def save(self, index: NumpyFlatIndex) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        np.savez(self._dir / "vectors.npz", **index.matrices)
        (self._dir / "meta.json").write_text(
            json.dumps({"ids": index.ids, "metas": index.metas}, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    def save_document(self, sha: str, data: bytes, suffix: str) -> Path:
        docs = self._dir / "docs"
        docs.mkdir(parents=True, exist_ok=True)
        path = docs / f"{sha}{suffix}"
        path.write_bytes(data)
        return path
