import numpy as np
from PIL import Image

import cv2


def _l2(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)


class SiglipEmbedder:
    """Texto e imagem no MESMO espaço vetorial (SigLIP2, ONNX, só onnxruntime: sem torch).

    É isso que permite buscar fotos com uma frase: o cosseno entre o vetor de "green dress" e o vetor de uma
    foto é alto quando a foto mostra um vestido verde. O DINOv2 do pipeline de documentos NÃO faz isso: ele
    mede aparência parecida entre duas imagens, mas não entende texto.

    Imagem: vira um quadrado (size x size) e é normalizada para [-1, 1]. `resize="pad"` preserva a proporção da
    foto com margem cinza; `"squash"` estica. Texto: minúsculas, truncado/preenchido em 64 tokens, sem moldura de
    frase ("a photo of a ...": nos testes não ajudou).
    """

    def __init__(
        self, vision_path, text_path, tokenizer_path, size: int = 224, resize: str = "pad", name: str = "siglip2",
        text_len: int = 64, batch: int = 8,
    ) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        if resize not in ("pad", "squash"):
            raise ValueError(f"resize deve ser 'pad' ou 'squash', não {resize!r}")
        providers = ["CPUExecutionProvider"]
        self._vision = ort.InferenceSession(str(vision_path), providers=providers)
        self._text = ort.InferenceSession(str(text_path), providers=providers)
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._tokenizer.enable_truncation(text_len)
        self._tokenizer.enable_padding(length=text_len, pad_id=0, pad_token="<pad>")
        self._size = size
        self._resize = resize
        self._batch = batch
        self.name = name  # identifica o espaço vetorial: vetores de modelos/pré-processamentos diferentes não se misturam

    def _prepare(self, image: np.ndarray) -> np.ndarray:
        rgb = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if self._resize == "pad":
            side = max(rgb.size)
            canvas = Image.new("RGB", (side, side), (128, 128, 128))
            canvas.paste(rgb, ((side - rgb.width) // 2, (side - rgb.height) // 2))
            rgb = canvas
        pixels = np.asarray(rgb.resize((self._size, self._size), Image.BILINEAR), dtype=np.float32) / 255.0
        return ((pixels - 0.5) / 0.5).transpose(2, 0, 1)

    def embed_images(self, images: list[np.ndarray]) -> np.ndarray:
        """(n, d), L2-normalizado."""
        chunks = []
        for start in range(0, len(images), self._batch):
            batch = np.stack([self._prepare(img) for img in images[start : start + self._batch]])
            chunks.append(self._vision.run(["pooler_output"], {"pixel_values": batch})[0])
        return _l2(np.vstack(chunks).astype(np.float32))

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """(n, d), L2-normalizado."""
        encoded = self._tokenizer.encode_batch([t.lower() for t in texts])
        ids = np.array([e.ids for e in encoded], dtype=np.int64)
        return _l2(self._text.run(["pooler_output"], {"input_ids": ids})[0].astype(np.float32))
