import base64

import numpy as np

from app.api.domain.bbox import BBox
from app.api.domain.entities import Word
from app.api.infrastructure.image_io import encode_png

_TOOL = {
    "name": "report_words",
    "description": "Registra cada palavra visível no documento com sua caixa delimitadora.",
    "input_schema": {
        "type": "object",
        "properties": {
            "words": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "x0": {"type": "number"},
                        "y0": {"type": "number"},
                        "x1": {"type": "number"},
                        "y1": {"type": "number"},
                        "confidence": {"type": "number"},
                    },
                    "required": ["text", "x0", "y0", "x1", "y1", "confidence"],
                },
            }
        },
        "required": ["words"],
    },
}

_PROMPT = (
    "Faça OCR literal deste comprovante. Liste TODAS as palavras, na ordem de leitura, sem corrigir "
    "nem normalizar nada (mantenha dígitos e pontuação exatamente como estão). Para cada palavra informe "
    "a caixa delimitadora em coordenadas normalizadas de 0 a 1000 (x0,y0 = canto superior esquerdo; "
    "x1,y1 = canto inferior direito) e sua confiança entre 0 e 1."
)


class LlmOcrEngine:
    """OCR via Claude vision.

    Bboxes de LLM são aproximadas (menos precisas que Tesseract/PP-OCR), então o IoU entre documentos
    é mais ruidoso. A confiança é auto-reportada pelo modelo, não uma probabilidade calibrada.
    """

    name = "llm"

    def __init__(self, api_key: str | None, model: str) -> None:
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def read(self, image: np.ndarray) -> list[Word]:
        h, w = image.shape[:2]
        data = base64.standard_b64encode(encode_png(image)).decode()
        response = self._client.messages.create(
            model=self._model,
            max_tokens=16000,
            tools=[_TOOL],
            tool_choice={"type": "tool", "name": "report_words"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}},
                        {"type": "text", "text": _PROMPT},
                    ],
                }
            ],
        )
        block = next(b for b in response.content if b.type == "tool_use")
        words = []
        for item in block.input["words"]:
            box = BBox(item["x0"] / 1000 * w, item["y0"] / 1000 * h, item["x1"] / 1000 * w, item["y1"] / 1000 * h)
            words.append(Word(item["text"].strip(), box, float(item["confidence"])))
        return words
