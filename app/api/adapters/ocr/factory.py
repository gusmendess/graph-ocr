from app.api.adapters.ocr.base import OcrEngine
from app.api.adapters.ocr.tesseract import find_tesseract
from app.api.settings import Settings


def build_engine(settings: Settings) -> OcrEngine:
    name = settings.ocr_engine.lower()
    if name == "auto":
        name = "tesseract" if find_tesseract(settings.tesseract_cmd) else "rapidocr"

    if name == "tesseract":
        from app.api.adapters.ocr.tesseract import TesseractEngine

        return TesseractEngine(settings.tesseract_cmd, settings.tesseract_lang)
    if name == "rapidocr":
        from app.api.adapters.ocr.rapidocr_engine import RapidOcrEngine

        return RapidOcrEngine()
    if name == "llm":
        from app.api.adapters.ocr.llm import LlmOcrEngine

        return LlmOcrEngine(settings.anthropic_api_key, settings.llm_model)
    raise ValueError(f"Motor de OCR desconhecido: {settings.ocr_engine!r}")
