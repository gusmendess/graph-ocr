import logging
from pathlib import Path

from app.api.services.document_embedder import DocumentEmbedder
from app.api.services.embedding_service import CharNgramEmbedder
from app.api.services.siglip_presets import SIGLIP_PRESETS
from app.api.services.text_image_embedder import SiglipEmbedder
from app.api.services.visual_embedder import VisualEmbedder, ensure_model
from app.api.settings import Settings

logger = logging.getLogger(__name__)


def build_document_embedder(settings: Settings) -> DocumentEmbedder:
    text = CharNgramEmbedder(settings.embed_dim, (settings.embed_ngram_min, settings.embed_ngram_max))
    visual = None
    if settings.visual_enabled:
        try:
            if not settings.visual_model_path.exists() and not settings.visual_auto_download:
                raise FileNotFoundError(settings.visual_model_path)
            model = ensure_model(settings.visual_model_path, settings.visual_model_url)
            visual = VisualEmbedder(model, settings.visual_size)
        except Exception as exc:  # sem rede/modelo: segue só com texto + layout, mas avisa
            logger.warning("Bloco visual desativado (%s: %s)", type(exc).__name__, exc)
    return DocumentEmbedder(text, visual, settings.layout_grid)


def build_siglip_embedder(settings: Settings) -> SiglipEmbedder:
    """SigLIP2 (texto + imagem no mesmo espaço), sem quantização. Baixa os arquivos que faltarem na primeira vez."""
    if settings.siglip_model not in SIGLIP_PRESETS:
        raise ValueError(f"siglip_model deve ser um de {sorted(SIGLIP_PRESETS)}, não {settings.siglip_model!r}")
    preset = SIGLIP_PRESETS[settings.siglip_model]
    root = settings.siglip_models_dir / preset.folder
    base = f"https://huggingface.co/{preset.repo}/resolve/main"

    def fetch(file: str):
        return ensure_model(root / Path(file).name, f"{base}/{file}")

    for extra in preset.extra:  # pesos externos ficam ao lado do .onnx
        fetch(extra)
    return SiglipEmbedder(
        fetch(preset.vision), fetch(preset.text), fetch("tokenizer.json"), size=preset.size,
        resize=settings.siglip_resize, name=f"siglip2-{settings.siglip_model}-{settings.siglip_resize}",
    )
