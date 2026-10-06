import re
from datetime import datetime, timedelta

from app.api.domain.entities import Word
from app.api.models.schemas import Entity, Flag

_PATTERNS = {
    "pix_e2e": re.compile(r"E\d{20}[A-Za-z0-9]{11}"),
    "cnpj": re.compile(r"\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}"),
    "cpf": re.compile(r"\d{3}\.\d{3}\.\d{3}-\d{2}"),
}
_UUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
_E2E_TS = re.compile(r"E\d{8}(\d{12})[A-Za-z0-9]{11}")
_DATE = re.compile(r"(?<!\d)(\d{2})/(\d{2})/(\d{4})(?!\d)")  # aceita texto colado ("2021às")
_TIME = re.compile(r"(?<!\d)(\d{2}):(\d{2})(?::(\d{2}))?(?!\d)")
_BRT = timedelta(hours=-3)


def extract_entities(words: list[Word]) -> list[Entity]:
    """Identificadores que ligam comprovantes entre si no grafo do caso."""
    text = " ".join(w.text for w in words)
    found: dict[tuple[str, str], Entity] = {}
    for kind, pattern in _PATTERNS.items():
        for value in pattern.findall(text):
            found[(kind, value)] = Entity(kind=kind, value=value)
    # chaves UUID costumam quebrar de linha no comprovante: junta tudo sem espaços
    for value in _UUID.findall(text.replace(" ", "")):
        found[("uuid", value.lower())] = Entity(kind="uuid", value=value.lower())
    return list(found.values())


def _first(words: list[Word], pattern: re.Pattern, start: int = 0) -> tuple[int, re.Match] | None:
    for i in range(start, len(words)):
        found = pattern.search(words[i].text)
        if found:
            return i, found
    return None


def consistency_flags(words: list[Word]) -> list[Flag]:
    """Regras sem template: a data exibida precisa bater com o timestamp embutido no ID Pix (E2E).

    O endToEndId Pix carrega yyyyMMddHHmm em UTC. Assumimos exibição em horário de Brasília (UTC-3).
    Cada Flag aponta as palavras envolvidas (`word_indexes`) para a visualização destacá-las.
    """
    e2e = _first(words, _E2E_TS)
    date = _first(words, _DATE)
    if not (e2e and date):
        return []
    time = _first(words, _TIME, date[0] + 1)
    stamp = datetime.strptime(e2e[1].group(1), "%Y%m%d%H%M") + _BRT
    d = date[1]
    shown_day = f"{d.group(1)}/{d.group(2)}/{d.group(3)}"

    flags = []
    if shown_day != stamp.strftime("%d/%m/%Y"):
        flags.append(
            Flag(
                code="PIX_DATE_MISMATCH",
                severity="high",
                detail=f"data exibida {shown_day} difere da data no ID Pix ({stamp:%d/%m/%Y})",
                word_indexes=[date[0], e2e[0]],
            )
        )
    if time:
        t = time[1]
        if f"{t.group(1)}:{t.group(2)}" != stamp.strftime("%H:%M"):
            flags.append(
                Flag(
                    code="PIX_TIME_MISMATCH",
                    severity="high",
                    detail=f"hora exibida {t.group(1)}:{t.group(2)} difere da hora no ID Pix ({stamp:%H:%M}, UTC-3)",
                    word_indexes=[time[0], e2e[0]],
                )
            )
    return flags
