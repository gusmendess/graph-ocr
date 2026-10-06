from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.api.domain.alignment import AlignmentConfig


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="GRAPH_OCR_", extra="ignore")

    # OCR
    ocr_engine: str = "auto"  # auto | tesseract | rapidocr | llm
    tesseract_cmd: str | None = None
    tesseract_lang: str = "por+eng"
    llm_model: str = "claude-sonnet-5-5"
    anthropic_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("ANTHROPIC_API_KEY", "GRAPH_OCR_ANTHROPIC_API_KEY")
    )
    cache_dir: Path = Path(".cache/ocr")
    base_image: Path = Path("image.png")  # referência usada por `make run <imagem>`

    # Cascata
    phash_threshold: float = 0.7  # abaixo disso o próprio phash decide (documentos diferentes)
    matcher: str = "vector"  # vector (cosseno entre vetores de texto + contexto do grafo) | iou (texto exato + IoU)
    embed_dim: int = 2048
    embed_ngram_min: int = 2
    embed_ngram_max: int = 4
    cos_same: float = 0.995  # cosseno >= isso: mesmo conteúdo
    cos_edit: float = 0.3  # entre cos_edit e cos_same: "quase igual" = edição no lugar; abaixo: sem par
    slot_overlap: float = 0.5  # Jaccard dos vizinhos-âncora >= isso: mesmo campo, mesmo com o texto trocado por inteiro
    context_weight: float = 0.3  # peso do contexto do grafo (vizinhos k-NN); 0 = só vetor de texto + cosseno
    ocr_different_threshold: float = 0.3  # score de OCR abaixo disso: conteúdo diferente (outro documento)
    iou_match: float = 0.5  # IoU mínimo para considerar a palavra "no mesmo lugar"
    iou_slot: float = 0.3  # IoU mínimo para dois textos diferentes ocuparem o mesmo "slot"
    knn_k: int = 4  # vizinhos por nó no grafo de palavras

    # Alinhamento de grafos (ver AlignmentConfig)
    alignment_model: str = "similarity"  # similarity | affine
    alignment_iterations: int = 2000
    alignment_inlier_tolerance: float = 0.4
    alignment_min_inliers: int = 4
    alignment_candidates_per_node: int = 5
    alignment_text_weight: float = 0.5
    alignment_max_scale: float = 4.0
    alignment_min_separation: float = 3.0
    alignment_seed: int = 0

    def alignment_config(self) -> AlignmentConfig:
        return AlignmentConfig(
            model=self.alignment_model,
            iterations=self.alignment_iterations,
            inlier_tolerance=self.alignment_inlier_tolerance,
            min_inliers=self.alignment_min_inliers,
            candidates_per_node=self.alignment_candidates_per_node,
            text_weight=self.alignment_text_weight,
            max_scale=self.alignment_max_scale,
            min_separation=self.alignment_min_separation,
            seed=self.alignment_seed,
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
