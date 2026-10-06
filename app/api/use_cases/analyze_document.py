from dataclasses import dataclass

import networkx as nx
import numpy as np

from app.api.domain.entities import Word
from app.api.domain.enums import DocumentVerdict
from app.api.infrastructure.image_io import decode_image
from app.api.models.schemas import DocumentReport
from app.api.services import entity_service, forensics_service, graph_service
from app.api.services.ocr_service import OcrService
from app.api.settings import Settings
from app.api.utils import sha256_bytes


@dataclass
class AnalyzeOutcome:
    report: DocumentReport
    image: np.ndarray
    words: list[Word]
    graph: nx.Graph


class AnalyzeDocumentUseCase:
    """Analisa UM documento, sem referência: OCR + regras de consistência que dispensam template.

    O veredito vem só das regras objetivas (ex.: data exibida x timestamp do ID Pix). A anomalia
    geométrica é reportada como sinal fraco e NÃO entra no veredito: é ruidosa em fontes proporcionais.
    `NO_EVIDENCE` significa que nenhuma regra foi violada, não que o documento é autêntico.
    """

    def __init__(self, settings: Settings, ocr: OcrService) -> None:
        self._settings = settings
        self._ocr = ocr

    def execute(self, data: bytes) -> AnalyzeOutcome:
        image = decode_image(data)
        sha = sha256_bytes(data)
        words = self._ocr.read(sha, image)
        graph = graph_service.build_word_graph(words, self._settings.knn_k)
        flags = entity_service.consistency_flags(words)

        if flags:
            verdict = DocumentVerdict.SUSPICIOUS
            summary = "; ".join(f"{f.code}: {f.detail}" for f in flags)
        else:
            verdict = DocumentVerdict.NO_EVIDENCE
            summary = "Nenhuma regra de consistência violada (isso não prova autenticidade)."

        report = DocumentReport(
            sha256=sha,
            n_words=len(words),
            entities=entity_service.extract_entities(words),
            flags=flags,
            anomalies=forensics_service.document_anomalies(graph),
            verdict=verdict,
            summary=summary,
            ocr_engine=self._ocr.engine_name,
        )
        return AnalyzeOutcome(report, image, words, graph)
