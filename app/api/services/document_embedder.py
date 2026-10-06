import numpy as np

from app.api.domain.entities import Word
from app.api.domain.vectors import DocumentVector
from app.api.services.embedding_service import Embedder
from app.api.services.layout_embedder import layout_vector
from app.api.services.vector_matching_service import group_lines


class DocumentEmbedder:
    """Vetor multimodal do documento: visual + texto + layout (cada bloco L2-normalizado).

    - visual:  aparência (DINOv2 ONNX). Opcional: sem modelo, o bloco simplesmente não existe.
    - texto:   conteúdo completo do OCR, com n-gramas de caracteres (mantém a diferença entre dígitos).
    - layout:  onde o texto fica na página (grade sobre o bbox do conteúdo; invariante a escala/translação).
    """

    def __init__(self, text: Embedder, visual=None, layout_grid: int = 16) -> None:
        self._text = text
        self._visual = visual
        self._grid = layout_grid

    @property
    def blocks(self) -> tuple[str, ...]:
        return ("visual", "text", "layout") if self._visual is not None else ("text", "layout")

    def embed(self, image: np.ndarray, words: list[Word]) -> DocumentVector:
        full_text = "\n".join(line.word.text for line in group_lines(words))
        blocks = {
            "text": self._text.embed([full_text])[0],
            "layout": layout_vector(words, self._grid),
        }
        if self._visual is not None:
            blocks["visual"] = self._visual.embed(image)
        return DocumentVector(blocks)
