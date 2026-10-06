from app.api.domain.entities import Word
from app.api.domain.enums import WordStatus
from app.api.models.schemas import WordMatch
from app.api.services.vector_matching_service import group_lines

_KEPT = (WordStatus.MATCHED, WordStatus.MODIFIED, WordStatus.MOVED)


def mark_cropped(matches: list[WordMatch], words_a: list[Word]) -> int:
    """Marca como CROPPED as palavras de A ausentes em B que formam um bloco contíguo no topo ou no rodapé.

    Print parcial é comum: B mostra só uma faixa do comprovante e o que está fora dela some de uma vez,
    na borda do documento. Isso não é edição. Já uma linha ausente NO MEIO do documento (cercada por
    linhas que casaram) continua REMOVED. Exige ao menos uma linha casada, senão seria outro documento.
    Devolve quantas palavras foram marcadas.
    """
    kept = {m.a_index for m in matches if m.status in _KEPT and m.a_index is not None}
    lines = group_lines(words_a)  # em ordem de leitura (topo -> rodapé)
    survives = [any(i in kept for i in line.indices) for line in lines]
    if not any(survives):
        return 0

    first, last = survives.index(True), len(survives) - 1 - survives[::-1].index(True)
    edge = {i for n, line in enumerate(lines) if n < first or n > last for i in line.indices}

    count = 0
    for m in matches:
        if m.status == WordStatus.REMOVED and m.a_index in edge:
            m.status = WordStatus.CROPPED
            count += 1
    return count
