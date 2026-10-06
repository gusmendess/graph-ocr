"""CLI: indexa comprovantes (vetor multimodal visual + texto + layout) para a busca reversa.

    uv run python scripts/index.py data/samples            # uma pasta (png/jpg/jpeg)
    uv run python scripts/index.py image.png image6.jpeg   # arquivos
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.api.adapters.ocr.factory import build_engine  # noqa: E402
from app.api.repositories.index_repository import DocumentIndexRepository  # noqa: E402
from app.api.repositories.ocr_cache import OcrCacheRepository  # noqa: E402
from app.api.services.embedder_factory import build_document_embedder  # noqa: E402
from app.api.services.ocr_service import OcrService  # noqa: E402
from app.api.settings import get_settings  # noqa: E402
from app.api.use_cases.search_documents import SearchDocumentsUseCase  # noqa: E402

_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def expand(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        files += sorted(p for p in path.iterdir() if p.suffix.lower() in _EXTENSIONS) if path.is_dir() else [path]
    return files


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", type=Path, help="arquivos ou pastas")
    parser.add_argument("--engine", help="auto | tesseract | rapidocr | llm")
    args = parser.parse_args(argv)

    settings = get_settings()
    if args.engine:
        settings = settings.model_copy(update={"ocr_engine": args.engine})
    files = expand(args.paths)
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        parser.error(f"arquivo não encontrado: {', '.join(missing)}")

    use_case = SearchDocumentsUseCase(
        settings,
        OcrService(build_engine(settings), OcrCacheRepository(settings.cache_dir)),
        build_document_embedder(settings),
        DocumentIndexRepository(settings.index_dir),
    )
    for file in files:
        result = use_case.add(file.read_bytes(), file.name, persist=False)
        print(f"{'atualizado' if result.replaced else 'indexado  '} {file.name}  blocos={','.join(result.blocks)}")
    use_case.save()
    print(f"índice: {use_case.n_indexed} documento(s) em {settings.index_dir}")


if __name__ == "__main__":
    main()
