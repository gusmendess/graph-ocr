import logging
import urllib.request
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def ensure_model(path: Path, url: str) -> Path:
    """Garante o arquivo ONNX local; baixa na primeira vez (escrita atômica, sem deixar arquivo pela metade)."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Baixando modelo visual de %s", url)
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, tmp)  # noqa: S310 (URL vem da configuração)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


class VisualEmbedder:
    """Aparência do documento (banco, cores, logo, estrutura) com DINOv2 em ONNX; devolve o token CLS (384-d).

    DINOv2 em vez de CLIP: é bem mais forte em recuperação de imagem quase duplicada e em estrutura;
    o CLIP é orientado a semântica ("um recibo") e põe todo comprovante no mesmo ponto.
    A imagem entra inteira, proporcional e com margem branca (letterbox): cortar o centro perderia conteúdo.
    """

    def __init__(self, model_path: Path, size: int = 448) -> None:
        import onnxruntime as ort

        self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self._input = self._session.get_inputs()[0].name
        self._size = size - size % 14  # o ViT usa patches de 14 px

    def embed(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        scale = self._size / max(h, w)
        resized = cv2.resize(image, (max(int(w * scale), 1), max(int(h * scale), 1)), interpolation=cv2.INTER_AREA)
        canvas = np.full((self._size, self._size, 3), 255, np.uint8)
        top, left = (self._size - resized.shape[0]) // 2, (self._size - resized.shape[1]) // 2
        canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = ((rgb - _MEAN) / _STD).transpose(2, 0, 1)[None]
        cls = self._session.run(None, {self._input: tensor})[0][0, 0]
        return (cls / max(float(np.linalg.norm(cls)), 1e-12)).astype(np.float32)
