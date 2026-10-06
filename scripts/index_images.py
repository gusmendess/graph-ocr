"""CLI: indexa fotos para a busca por linguagem natural (SigLIP2: texto e imagem no mesmo espaço vetorial).

    uv run python scripts/index_images.py                  # a pasta padrão (GRAPH_OCR_IMAGES_SOURCE)
    uv run python scripts/index_images.py <pasta>          # uma pasta
    uv run python scripts/index_images.py <foto> ...       # arquivos
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.api.infrastructure.image_files import expand_images  # noqa: E402
from app.api.repositories.semantic_index_repository import SemanticIndexRepository  # noqa: E402
from app.api.services.embedder_factory import build_siglip_embedder  # noqa: E402
from app.api.settings import get_settings  # noqa: E402
from app.api.use_cases.search_images import SearchImagesUseCase  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", type=Path, help="pastas ou fotos (padrão: GRAPH_OCR_IMAGES_SOURCE)")
    args = parser.parse_args()

    settings = get_settings()
    try:
        files = expand_images(args.paths or [settings.images_source])
    except FileNotFoundError as exc:
        parser.error(str(exc))
    if not files:
        parser.error("nenhuma imagem encontrada (extensões: png, jpg, jpeg, webp, bmp)")

    try:
        use_case = SearchImagesUseCase(
            settings, build_siglip_embedder(settings), SemanticIndexRepository(settings.semantic_index_dir)
        )
    except ValueError as exc:
        sys.exit(str(exc))
    for result in use_case.add_many([(f.read_bytes(), f.name) for f in files]):
        print(f"{'atualizado' if result.replaced else 'indexado  '} {result.label}")
    print(f"índice de imagens: {use_case.n_indexed} imagem(ns) em {settings.semantic_index_dir}")


if __name__ == "__main__":
    main()
