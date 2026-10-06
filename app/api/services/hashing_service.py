import cv2
import numpy as np

from app.api.utils import sha256_bytes

__all__ = ["sha256_bytes", "phash", "phash_thumbnail", "phash_similarity", "phash_to_hex", "phash_from_hex"]


def phash_thumbnail(image: np.ndarray, hash_size: int = 8, highfreq: int = 4) -> np.ndarray:
    """A miniatura em tons de cinza (32x32) que o phash realmente enxerga."""
    side = hash_size * highfreq
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return cv2.resize(gray, (side, side), interpolation=cv2.INTER_AREA)


def phash(image: np.ndarray, hash_size: int = 8, highfreq: int = 4) -> np.ndarray:
    """pHash clássico: DCT de uma miniatura 32x32; bit = coeficiente acima da mediana."""
    small = phash_thumbnail(image, hash_size, highfreq).astype(np.float32)
    low = cv2.dct(small)[:hash_size, :hash_size]
    return (low > np.median(low.flatten()[1:])).flatten()  # ignora o coeficiente DC


def phash_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """1 - distância de Hamming normalizada."""
    return 1.0 - float(np.count_nonzero(a != b)) / a.size


def phash_to_hex(bits: np.ndarray) -> str:
    """Os 64 bits do phash em 16 caracteres hexadecimais (para guardar junto do documento no índice)."""
    return np.packbits(bits.astype(np.uint8)).tobytes().hex()


def phash_from_hex(text: str) -> np.ndarray:
    return np.unpackbits(np.frombuffer(bytes.fromhex(text), dtype=np.uint8)).astype(bool)
