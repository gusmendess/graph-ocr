import cv2
import numpy as np

from app.api.utils import sha256_bytes

__all__ = ["sha256_bytes", "phash", "phash_similarity"]


def phash(image: np.ndarray, hash_size: int = 8, highfreq: int = 4) -> np.ndarray:
    """pHash clássico: DCT de uma miniatura 32x32; bit = coeficiente acima da mediana."""
    side = hash_size * highfreq
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    small = cv2.resize(gray, (side, side), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(small)[:hash_size, :hash_size]
    return (low > np.median(low.flatten()[1:])).flatten()  # ignora o coeficiente DC


def phash_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """1 - distância de Hamming normalizada."""
    return 1.0 - float(np.count_nonzero(a != b)) / a.size
