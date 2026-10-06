from functools import lru_cache

from fastapi import APIRouter, Depends, File, HTTPException, Query, Response, UploadFile

from app.api.adapters.ocr.factory import build_engine
from app.api.models.schemas import CascadeResult
from app.api.repositories.ocr_cache import OcrCacheRepository
from app.api.services.ocr_service import OcrService
from app.api.services.visualization_service import render_comparison
from app.api.settings import get_settings
from app.api.use_cases.compare_documents import CompareDocumentsUseCase

router = APIRouter(prefix="/compare", tags=["compare"])


@lru_cache
def get_use_case() -> CompareDocumentsUseCase:
    settings = get_settings()
    ocr = OcrService(build_engine(settings), OcrCacheRepository(settings.cache_dir))
    return CompareDocumentsUseCase(settings, ocr)


async def _run(a: UploadFile, b: UploadFile, full: bool, use_case: CompareDocumentsUseCase):
    try:
        return use_case.execute(
            await a.read(), await b.read(), a.filename or "A", b.filename or "B", run_all=full
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("", response_model=CascadeResult)
async def compare(
    a: UploadFile = File(..., description="documento de referência"),
    b: UploadFile = File(..., description="documento suspeito"),
    full: bool = Query(False, description="executa todas as camadas da cascata"),
    use_case: CompareDocumentsUseCase = Depends(get_use_case),
):
    return (await _run(a, b, full, use_case)).result


@router.post("/visual", response_class=Response)
async def compare_visual(
    a: UploadFile = File(...),
    b: UploadFile = File(...),
    use_case: CompareDocumentsUseCase = Depends(get_use_case),
):
    outcome = await _run(a, b, True, use_case)
    png = render_comparison(outcome, a.filename or "A", b.filename or "B")
    return Response(content=png, media_type="image/png")
