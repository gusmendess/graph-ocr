from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="GRAPH_OCR_", extra="ignore")

    # OCR (alimenta os blocos de texto e de layout do vetor)
    ocr_engine: str = "auto"  # auto | tesseract | rapidocr | llm
    tesseract_cmd: str | None = None
    tesseract_lang: str = "por+eng"
    llm_model: str = "claude-sonnet-5-5"
    anthropic_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("ANTHROPIC_API_KEY", "GRAPH_OCR_ANTHROPIC_API_KEY")
    )
    cache_dir: Path = Path(".cache/ocr")
    index_source: Path = Path("data/samples/documents")  # o que `make run-vector` e `make index` indexam por padrão
    images_source: Path = Path("data/samples/dresses")  # o que `make index-images` indexa por padrão

    # Cascata: sha256 -> phash -> vetor
    phash_threshold: float = 0.3  # abaixo disso o próprio phash decide (o par nem chega ao OCR e ao vetor); 0 desliga o atalho

    # Vetor multimodal (visual + texto + layout)
    embed_dim: int = 2048  # bloco de texto: n-gramas de caracteres em hash
    embed_ngram_min: int = 2
    embed_ngram_max: int = 4
    weight_visual: float = 0.01  # pesos do score composto; calibrar com pares reais (original, fraude, outro banco)
    weight_text: float = 0.90
    weight_layout: float = 0.09
    visual_enabled: bool = True
    visual_auto_download: bool = True  # baixa o ONNX na primeira vez; sem rede, o bloco visual é desativado
    visual_model_path: Path = Path(".cache/models/dinov2-small.onnx")
    visual_model_url: str = "https://huggingface.co/onnx-community/dinov2-small/resolve/main/onnx/model.onnx"
    visual_size: int = 448  # lado do quadrado (letterbox) que entra no DINOv2; múltiplo de 14
    layout_grid: int = 16
    index_dir: Path = Path(".cache/index")
    search_k: int = 10

    # Busca de imagens por linguagem natural: SigLIP2 em ONNX (texto e imagem no mesmo espaço vetorial)
    # Sem quantização: o int8 degradou o ranking (mAP 0.945 -> 1.000 ao tirar; no modelo grande foi pior que o pequeno).
    siglip_model: str = "base"  # base (1,5 GB, 0,13 s/imagem) | so400m (4,5 GB, 2,5 s/imagem, mais robusto)
    siglip_resize: str = "pad"  # pad = preserva a proporção da foto (margem cinza) | squash = estica para um quadrado
    siglip_models_dir: Path = Path(".cache/models")
    semantic_index_dir: Path = Path(".cache/semantic_index")

    def search_weights(self) -> dict[str, float]:
        return {"visual": self.weight_visual, "text": self.weight_text, "layout": self.weight_layout}


@lru_cache
def get_settings() -> Settings:
    return Settings()
