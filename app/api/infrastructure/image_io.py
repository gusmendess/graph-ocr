import cv2
import numpy as np


def decode_image(data: bytes) -> np.ndarray:
    """Bytes -> imagem BGR (uint8)."""
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Não foi possível decodificar a imagem")
    return img


def encode_png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("Falha ao codificar PNG")
    return buf.tobytes()
