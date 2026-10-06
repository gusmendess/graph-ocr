import logging

from app.api.services.document_embedder import DocumentEmbedder
from app.api.services.embedding_service import CharNgramEmbedder
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
