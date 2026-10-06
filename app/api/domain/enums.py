from enum import Enum


class Verdict(str, Enum):
    IDENTICAL = "identical"  # mesmos bytes (sha256)
    DIFFERENT_DOCUMENT = "different_document"  # phash abaixo do limiar: nem chega ao OCR
    EQUIVALENT = "equivalent"  # pixels diferentes, mas mesmo conteúdo e posição (ex.: recompressão)
    TAMPERED = "tampered"  # palavras alteradas, inseridas ou removidas
    CROPPED = "cropped"  # a suspeita é só um recorte da base (topo/rodapé ausentes); o que aparece é igual


class DocumentVerdict(str, Enum):
    """Veredito de um documento isolado (sem referência): só regras objetivas decidem."""

    SUSPICIOUS = "suspicious"  # alguma regra de consistência foi violada
    NO_EVIDENCE = "no_evidence"  # nenhuma regra violada; NÃO prova autenticidade


class WordStatus(str, Enum):
    MATCHED = "matched"  # mesmo texto, mesma posição (IoU alto)
    MOVED = "moved"  # mesmo texto, posição deslocada
    MODIFIED = "modified"  # mesma posição, texto diferente: edição clássica
    REMOVED = "removed"  # só existe no documento A
    CROPPED = "cropped"  # só em A, num bloco contíguo na borda: ficou fora do recorte de B
    ADDED = "added"  # só existe no documento B


class FieldKind(str, Enum):
    DATE = "date"
    TIME = "time"
    MONEY = "money"
    ID = "id"
    TEXT = "text"
