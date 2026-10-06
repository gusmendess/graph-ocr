from functools import lru_cache

from app.api.adapters.ocr.factory import build_engine
from app.api.repositories.index_repository import DocumentIndexRepository
from app.api.repositories.ocr_cache import OcrCacheRepository
from app.api.repositories.semantic_index_repository import SemanticIndexRepository
from app.api.services.embedder_factory import build_document_embedder, build_siglip_embedder
from app.api.services.ocr_service import OcrService
from app.api.settings import get_settings
from app.api.use_cases.search_documents import SearchDocumentsUseCase
from app.api.use_cases.search_images import SearchImagesUseCase


@lru_cache
def get_ocr_service() -> OcrService:
    settings = get_settings()
    return OcrService(build_engine(settings), OcrCacheRepository(settings.cache_dir))


@lru_cache
def get_image_search_use_case() -> SearchImagesUseCase:
    settings = get_settings()
    return SearchImagesUseCase(
        settings, build_siglip_embedder(settings), SemanticIndexRepository(settings.semantic_index_dir)
    )


@lru_cache
def get_search_use_case() -> SearchDocumentsUseCase:
    settings = get_settings()
    return SearchDocumentsUseCase(
        settings, get_ocr_service(), build_document_embedder(settings), DocumentIndexRepository(settings.index_dir)
    )
