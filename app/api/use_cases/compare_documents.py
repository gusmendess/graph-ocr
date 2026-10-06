import time
from dataclasses import dataclass

import networkx as nx
import numpy as np

from app.api.domain.alignment import Alignment
from app.api.domain.entities import Word
from app.api.domain.enums import Verdict, WordStatus
from app.api.infrastructure.image_io import decode_image
from app.api.models.schemas import AlignmentInfo, CascadeResult, DocumentAnalysis, StageResult
from app.api.services import (
    alignment_service,
    crop_service,
    entity_service,
    forensics_service,
    graph_service,
    hashing_service,
    matching_service,
    vector_matching_service,
)
from app.api.services.embedding_service import CharNgramEmbedder, Embedder
from app.api.services.ocr_service import OcrService
from app.api.settings import Settings

_CHANGED = (WordStatus.MODIFIED, WordStatus.ADDED, WordStatus.REMOVED)


@dataclass
class CompareOutcome:
    """Resultado serializável + artefatos em memória que a visualização precisa."""

    result: CascadeResult
    image_a: np.ndarray
    image_b: np.ndarray
    words_a: list[Word] | None = None
    words_b: list[Word] | None = None
    graph_a: nx.Graph | None = None
    graph_b: nx.Graph | None = None
    alignment: Alignment | None = None


class CompareDocumentsUseCase:
    """Cascata: camadas baratas resolvem os casos óbvios; só os pares ambíguos chegam ao OCR.

    1. sha256: bytes idênticos -> score 1.0 (~0 ms)
    2. phash: similaridade perceptual; abaixo do limiar o próprio phash decide (sem OCR)
    3. OCR (texto + bbox + confiança) + grafo k-NN: pareia palavras por IoU/contexto e acha edições

    `run_all=True` executa todas as camadas mesmo após uma decisão (útil para demo/visualização);
    o veredito continua sendo o da primeira camada que decidiu.
    """

    def __init__(self, settings: Settings, ocr: OcrService, embedder: Embedder | None = None) -> None:
        self._settings = settings
        self._ocr = ocr
        self._embedder = embedder or CharNgramEmbedder(
            settings.embed_dim, (settings.embed_ngram_min, settings.embed_ngram_max)
        )

    def execute(
        self, data_a: bytes, data_b: bytes, label_a: str = "A", label_b: str = "B", run_all: bool = False
    ) -> CompareOutcome:
        img_a, img_b = decode_image(data_a), decode_image(data_b)
        stages: list[StageResult] = []
        verdict: Verdict | None = None
        decided_by = ""
        score = 0.0
        summary = ""

        # 1. sha256
        t0 = time.perf_counter()
        sha_a, sha_b = hashing_service.sha256_bytes(data_a), hashing_service.sha256_bytes(data_b)
        same = sha_a == sha_b
        stages.append(
            StageResult(
                name="sha256",
                executed=True,
                decided=same,
                score=1.0 if same else 0.0,
                elapsed_ms=_ms(t0),
                detail="bytes idênticos" if same else "bytes diferentes",
            )
        )
        if same:
            verdict, decided_by, score, summary = Verdict.IDENTICAL, "sha256", 1.0, "Arquivos idênticos."
            if not run_all:
                return self._finish(verdict, score, decided_by, summary, stages, img_a, img_b)

        # 2. phash
        t0 = time.perf_counter()
        sim = hashing_service.phash_similarity(hashing_service.phash(img_a), hashing_service.phash(img_b))
        different = sim < self._settings.phash_threshold
        stages.append(
            StageResult(
                name="phash",
                executed=True,
                decided=different and verdict is None,
                score=sim,
                elapsed_ms=_ms(t0),
                detail=f"similaridade perceptual {sim:.3f} (limiar {self._settings.phash_threshold})",
            )
        )
        if different and verdict is None:
            verdict, decided_by, score = Verdict.DIFFERENT_DOCUMENT, "phash", sim
            summary = "Documentos visualmente diferentes (phash abaixo do limiar); OCR não necessário."
            if not run_all:
                return self._finish(verdict, score, decided_by, summary, stages, img_a, img_b)

        # 3. OCR + grafo
        t0 = time.perf_counter()
        words_a, words_b = self._ocr.read(sha_a, img_a), self._ocr.read(sha_b, img_b)
        s = self._settings
        graph_a, graph_b = graph_service.build_word_graph(words_a, s.knn_k), graph_service.build_word_graph(words_b, s.knn_k)
        alignment = alignment_service.estimate_alignment(graph_a, graph_b, s.alignment_config(), s.knn_k)
        if s.matcher == "vector":
            matches = vector_matching_service.match_by_vectors(
                words_a, words_b, self._embedder, s.knn_k, s.cos_same, s.cos_edit, s.context_weight, s.slot_overlap
            )
        else:
            matches = matching_service.match_words(
                words_a, words_b, graph_a, graph_b, s.iou_match, s.iou_slot, alignment
            )
        n_cropped = crop_service.mark_cropped(matches, words_a)
        forensics_service.annotate_matches(matches, graph_b, alignment)
        ocr_score = matching_service.similarity_score(matches)

        ents_a, ents_b = entity_service.extract_entities(words_a), entity_service.extract_entities(words_b)
        doc_a = DocumentAnalysis(
            sha256=sha_a,
            n_words=len(words_a),
            entities=ents_a,
            flags=entity_service.consistency_flags(words_a),
            anomalies=forensics_service.document_anomalies(graph_a),
        )
        doc_b = DocumentAnalysis(
            sha256=sha_b,
            n_words=len(words_b),
            entities=ents_b,
            flags=entity_service.consistency_flags(words_b),
            anomalies=forensics_service.document_anomalies(graph_b),
        )
        case_graph = graph_service.build_case_graph({label_a: ents_a, label_b: ents_b})

        changed = [m for m in matches if m.status in _CHANGED]
        tampered = bool(changed)  # as regras de consistência são só informativas: não decidem o veredito
        # No modo completo o OCR corrige o phash: só vale "documento diferente" se o conteúdo também difere.
        override_phash = run_all and verdict == Verdict.DIFFERENT_DOCUMENT
        stages.append(
            StageResult(
                name=f"ocr:{self._ocr.engine_name}",
                executed=True,
                decided=verdict is None or override_phash,
                score=ocr_score,
                elapsed_ms=_ms(t0),
                detail=(
                    f"{len(words_a)}/{len(words_b)} palavras; {len(changed)} alterada(s); "
                    f"{n_cropped} fora do recorte; {len(doc_a.flags) + len(doc_b.flags)} flag(s) de consistência; "
                    f"alinhamento B→A: {_describe(alignment)}"
                ),
            )
        )
        if verdict is None or override_phash:
            if ocr_score < s.ocr_different_threshold:
                verdict = Verdict.DIFFERENT_DOCUMENT
                summary = f"Conteúdo diferente (apenas {ocr_score:.0%} das palavras casam)."
            elif tampered:
                verdict = Verdict.TAMPERED
                summary = _summarize(changed, doc_a, doc_b) + _crop_note(n_cropped)
            elif n_cropped:
                verdict = Verdict.CROPPED
                summary = f"B é um recorte de A ({n_cropped} palavras fora do enquadramento, na borda); o que aparece é idêntico."
            else:
                verdict = Verdict.EQUIVALENT
                summary = "Mesmo conteúdo e posição; só os pixels diferem."
            if override_phash:
                for stage in stages:
                    if stage.name == "phash":
                        stage.decided = False  # o phash sinalizou "diferente", mas o OCR prevaleceu
                summary = f"(phash {stages[1].score:.2f} < limiar, mas o OCR prevaleceu) " + summary
            decided_by, score = f"ocr:{self._ocr.engine_name}", ocr_score

        result = CascadeResult(
            verdict=verdict,
            score=score,
            decided_by=decided_by,
            summary=summary,
            ocr_engine=self._ocr.engine_name,
            stages=stages,
            alignment=_alignment_info(alignment),
            matches=matches,
            doc_a=doc_a,
            doc_b=doc_b,
            shared_entities=graph_service.shared_entities(case_graph),
        )
        return CompareOutcome(result, img_a, img_b, words_a, words_b, graph_a, graph_b, alignment)

    @staticmethod
    def _finish(verdict, score, decided_by, summary, stages, img_a, img_b) -> CompareOutcome:
        result = CascadeResult(verdict=verdict, score=score, decided_by=decided_by, summary=summary, stages=stages)
        return CompareOutcome(result, img_a, img_b)


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)


def _crop_note(n_cropped: int) -> str:
    return f"; (além disso, {n_cropped} palavras de A ficaram fora do recorte de B)" if n_cropped else ""


def _describe(al: Alignment) -> str:
    if not al.converged:
        return f"não convergiu ({al.candidates} candidatos); comparando sem transformação"
    tx, ty = al.translation
    return (
        f"{al.model}, escala {al.scale:.3f}, rotação {al.rotation_deg:+.2f}°, "
        f"deslocamento ({tx:+.1f}, {ty:+.1f}) px, {al.inliers} nós sobrepostos"
    )


def _alignment_info(al: Alignment) -> AlignmentInfo:
    tx, ty = al.translation
    return AlignmentInfo(
        model=al.model,
        converged=al.converged,
        inliers=al.inliers,
        candidates=al.candidates,
        scale=al.scale,
        rotation_deg=al.rotation_deg,
        tx=tx,
        ty=ty,
        matrix=al.matrix.tolist(),
    )


def _summarize(changed, doc_a: DocumentAnalysis, doc_b: DocumentAnalysis) -> str:
    parts = []
    for m in changed:
        a = m.a.text if m.a else "∅"
        b = m.b.text if m.b else "∅"
        cos = f", cos {m.cosine:.2f}" if m.cosine is not None else ""
        parts.append(f"{m.status.value} [{m.kind.value}] '{a}' -> '{b}' (suspeita {m.suspicion:.2f}{cos})")
    parts += [f"[informativo] {f.code}: {f.detail}" for f in doc_a.flags + doc_b.flags]
    return "; ".join(parts)
