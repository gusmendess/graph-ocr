from __future__ import annotations

from pydantic import BaseModel, Field

from app.api.domain.entities import Word
from app.api.domain.enums import DocumentVerdict, FieldKind, Verdict, WordStatus


class WordDeltas(BaseModel):
    """Diferença geométrica entre a palavra em A e em B (B - A)."""

    height_pct: float
    char_width_pct: float
    baseline_px: float
    conf: float


class WordMatch(BaseModel):
    status: WordStatus
    a: Word | None = None
    b: Word | None = None
    a_index: int | None = None
    b_index: int | None = None
    iou: float = 0.0
    neighbor_jaccard: float = 0.0
    cosine: float | None = None  # similaridade de cosseno entre os vetores de texto (matcher=vector)
    kind: FieldKind = FieldKind.TEXT
    deltas: WordDeltas | None = None
    geometry_anomaly: float = 0.0  # quão destoante a palavra é dos vizinhos de linha (0-1)
    suspicion: float = 0.0  # heurística 0-1 (ver forensics_service)


class StageResult(BaseModel):
    name: str
    executed: bool
    decided: bool = False
    score: float | None = None
    elapsed_ms: float = 0.0
    detail: str = ""


class Entity(BaseModel):
    kind: str  # pix_e2e | cnpj | cpf | uuid
    value: str


class Flag(BaseModel):
    code: str
    severity: str  # low | medium | high
    detail: str
    word_indexes: list[int] = Field(default_factory=list)  # palavras do OCR envolvidas na regra


class AnomalyWord(BaseModel):
    text: str
    score: float
    index: int | None = None


class DocumentAnalysis(BaseModel):
    sha256: str
    n_words: int
    entities: list[Entity] = Field(default_factory=list)
    flags: list[Flag] = Field(default_factory=list)  # regras que dispensam template
    anomalies: list[AnomalyWord] = Field(default_factory=list)


class DocumentReport(DocumentAnalysis):
    """Análise de um documento isolado (sem referência)."""

    verdict: DocumentVerdict
    summary: str
    ocr_engine: str


class SharedEntity(BaseModel):
    kind: str
    value: str
    documents: list[str]


class AlignmentInfo(BaseModel):
    """Transformação B -> A estimada sobrepondo os grafos de palavras."""

    model: str
    converged: bool
    inliers: int
    candidates: int
    scale: float
    rotation_deg: float
    tx: float
    ty: float
    matrix: list[list[float]]


class CascadeResult(BaseModel):
    verdict: Verdict
    score: float
    decided_by: str
    summary: str
    ocr_engine: str | None = None
    stages: list[StageResult]
    alignment: AlignmentInfo | None = None
    matches: list[WordMatch] = Field(default_factory=list)
    doc_a: DocumentAnalysis | None = None
    doc_b: DocumentAnalysis | None = None
    shared_entities: list[SharedEntity] = Field(default_factory=list)


class SearchHit(BaseModel):
    doc_id: str
    label: str
    score: float  # cosseno composto (média ponderada dos cossenos de cada bloco)
    block_scores: dict[str, float]  # cosseno de cada bloco: visual / text / layout
    is_query: bool = False  # o documento consultado já estava no índice (mesmos bytes)


class SearchResponse(BaseModel):
    exact_match: bool  # sha256 idêntico a um documento indexado
    blocks_used: list[str]
    weights_used: dict[str, float]
    n_indexed: int
    hits: list[SearchHit]


class IndexResponse(BaseModel):
    doc_id: str
    label: str
    replaced: bool
    n_indexed: int
    blocks: list[str]
