from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from app.api.endpoints.deps import get_search_use_case
from app.api.models.schemas import IndexResponse, SearchResponse
from app.api.settings import get_settings
from app.api.use_cases.search_documents import SearchDocumentsUseCase

router = APIRouter(tags=["search"])


@router.post("/index", response_model=IndexResponse)
async def index_document(
    file: UploadFile = File(..., description="documento a indexar"),
    use_case: SearchDocumentsUseCase = Depends(get_search_use_case),
):
    try:
        return use_case.add(await file.read(), file.filename or "documento.png")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/search", response_model=SearchResponse)
async def search(
    file: UploadFile = File(..., description="documento consultado"),
    k: int = Query(5, ge=1, le=100),
    w_visual: float | None = Query(None, ge=0, description="peso do bloco visual"),
    w_text: float | None = Query(None, ge=0, description="peso do bloco de texto"),
    w_layout: float | None = Query(None, ge=0, description="peso do bloco de layout"),
    use_case: SearchDocumentsUseCase = Depends(get_search_use_case),
):
    weights = get_settings().search_weights()
    for name, value in (("visual", w_visual), ("text", w_text), ("layout", w_layout)):
        if value is not None:
            weights[name] = value
    try:
        return use_case.search(await file.read(), k, weights).response
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
