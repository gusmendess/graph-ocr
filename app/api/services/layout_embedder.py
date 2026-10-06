import numpy as np

from app.api.domain.entities import Word


def layout_vector(words: list[Word], grid: int = 16) -> np.ndarray:
    """Onde o texto fica na página: fração de cada célula de uma grade grid×grid coberta por texto.

    A grade é aplicada sobre o bbox do CONTEÚDO (união das palavras), então o vetor não depende da
    escala, da margem nem da translação do enquadramento. Determinístico e sem modelo.
    """
    out = np.zeros((grid, grid), dtype=np.float32)
    if not words:
        return out.flatten()

    x0 = min(w.bbox.x0 for w in words)
    y0 = min(w.bbox.y0 for w in words)
    x1 = max(w.bbox.x1 for w in words)
    y1 = max(w.bbox.y1 for w in words)
    width, height = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
    cell_w, cell_h = width / grid, height / grid
    edges_x = x0 + cell_w * np.arange(grid + 1)
    edges_y = y0 + cell_h * np.arange(grid + 1)

    for w in words:
        b = w.bbox
        overlap_x = np.clip(np.minimum(edges_x[1:], b.x1) - np.maximum(edges_x[:-1], b.x0), 0, None)
        overlap_y = np.clip(np.minimum(edges_y[1:], b.y1) - np.maximum(edges_y[:-1], b.y0), 0, None)
        out += np.outer(overlap_y, overlap_x) / (cell_w * cell_h)

    out = np.clip(out, 0.0, 1.0).flatten()
    norm = np.linalg.norm(out)
    return out / norm if norm > 0 else out
