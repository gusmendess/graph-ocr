from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from app.api.endpoints.deps import get_image_search_use_case
from app.api.models.schemas import ImageIndexResponse, ImageSearchResponse
from app.api.use_cases.search_images import SearchImagesUseCase

router = APIRouter(prefix="/images", tags=["images"])


@router.post("/index", response_model=ImageIndexResponse)
async def index_image(
    file: UploadFile = File(..., description="foto a indexar"),
    use_case: SearchImagesUseCase = Depends(get_image_search_use_case),
):
    try:
        return use_case.add(await file.read(), file.filename or "imagem.png")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/search", response_model=ImageSearchResponse)
def search_images(
    q: str = Query(..., min_length=1, description='frase em linguagem natural, ex.: "green dress"'),
    k: int = Query(10, ge=1, le=100),
    use_case: SearchImagesUseCase = Depends(get_image_search_use_case),
):
    try:
        return use_case.search_text(q, k).response
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
