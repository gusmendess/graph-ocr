from app.api.domain.entities import Word
from app.api.services.vector_matching_service import group_lines


def render_ocr_text(words: list[Word], label: str, engine: str) -> str:
    """Saída do OCR em texto puro, pensada para comparar duas imagens lado a lado (diff, editor).

    Duas seções: o texto linha a linha (ordem de leitura) e a tabela de palavras com confiança e bbox.
    """
    lines = group_lines(words)
    out = [
        f"# OCR de {label}",
        f"# motor: {engine} | linhas: {len(lines)} | palavras: {len(words)}",
        "",
        "## Linhas (ordem de leitura)",
        "  n | conf | texto",
    ]
    for n, line in enumerate(lines, 1):
        out.append(f"{n:3d} | {line.word.conf:.2f} | {line.word.text}")

    out += ["", "## Palavras", "  n | linha | conf | x0,y0,x1,y1 (px) | texto"]
    line_of = {i: n for n, line in enumerate(lines, 1) for i in line.indices}
    for i, w in enumerate(words):
        b = w.bbox
        out.append(f"{i:3d} | {line_of[i]:5d} | {w.conf:.2f} | {b.x0:.0f},{b.y0:.0f},{b.x1:.0f},{b.y1:.0f} | {w.text}")
    return "\n".join(out) + "\n"
