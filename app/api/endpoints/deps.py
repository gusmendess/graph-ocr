from functools import lru_cache

from app.api.adapters.ocr.factory import build_engine
from app.api.repositories.index_repository import DocumentIndexRepository
from app.api.repositories.ocr_cache import OcrCacheRepository
from app.api.services.embedder_factory import build_document_embedder
from app.api.services.ocr_service import OcrService
from app.api.settings import get_settings
from app.api.use_cases.compare_documents import CompareDocumentsUseCase
from app.api.use_cases.search_documents import SearchDocumentsUseCase


@lru_cache
def get_ocr_service() -> OcrService:
    settings = get_settings()
    return OcrService(build_engine(settings), OcrCacheRepository(settings.cache_dir))


@lru_cache
def get_compare_use_case() -> CompareDocumentsUseCase:
    return CompareDocumentsUseCase(get_settings(), get_ocr_service())


@lru_cache
def get_search_use_case() -> SearchDocumentsUseCase:
    settings = get_settings()
    return SearchDocumentsUseCase(
        settings, get_ocr_service(), build_document_embedder(settings), DocumentIndexRepository(settings.index_dir)
    )
