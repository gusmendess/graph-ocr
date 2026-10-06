from __future__ import annotations

from pydantic import BaseModel


class SearchHit(BaseModel):
    doc_id: str
    label: str
    score: float  # similaridade de 0 a 1 (ver `decided_by` para saber como foi calculada)
    decided_by: str  # sha256 (bytes idênticos) | phash (barrado: score = similaridade perceptual) | vector (cosseno)
    phash: float | None = None  # similaridade perceptual com a consulta (None quando o sha256 já decidiu)
    block_scores: dict[str, float] = {}  # cosseno de cada bloco (visual / text / layout); só quando decided_by = vector
    is_query: bool = False  # o documento consultado já estava no índice (mesmos bytes)


class SearchResponse(BaseModel):
    exact_match: bool  # sha256 idêntico a um documento indexado
    vector_computed: bool  # False se todos os pares foram decididos por sha256/phash (sem OCR e sem rede neural)
    phash_threshold: float  # abaixo disso o phash decide sozinho
    stage_counts: dict[str, int] = {}  # quantos pares cada estágio decidiu, entre TODOS os indexados (não só os k exibidos)
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


class ImageHit(BaseModel):
    rank: int
    doc_id: str
    label: str
    score: float  # cosseno entre o vetor da consulta e o da imagem (vale pela ordem do ranking, não como probabilidade)


class ImageSearchResponse(BaseModel):
    query: str
    n_indexed: int
    hits: list[ImageHit]


class ImageIndexResponse(BaseModel):
    doc_id: str
    label: str
    replaced: bool
    n_indexed: int
