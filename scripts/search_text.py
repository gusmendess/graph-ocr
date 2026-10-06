"""CLI: busca imagens por linguagem natural. As fotos precisam ter sido indexadas com index_images.py.

    uv run python scripts/search_text.py "green dress" -k 6
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.api.infrastructure.image_io import decode_image  # noqa: E402
from app.api.infrastructure.output_io import write_output  # noqa: E402
from app.api.repositories.semantic_index_repository import SemanticIndexRepository  # noqa: E402
from app.api.services.embedder_factory import build_siglip_embedder  # noqa: E402
from app.api.services.image_search_visualization_service import render_contact_sheet  # noqa: E402
from app.api.settings import get_settings  # noqa: E402
from app.api.use_cases.search_images import SearchImagesUseCase  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("query", help='frase em linguagem natural, ex.: "green dress"')
    parser.add_argument("-k", type=int, help="quantas imagens (padrão: GRAPH_OCR_SEARCH_K)")
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    parser.add_argument("--no-plots", action="store_true", help="não gera a folha de contato")
    args = parser.parse_args()

    if not args.query.strip():
        parser.error("a consulta está vazia (no make: make search-text Q=\"green dress\")")
    settings = get_settings()
    try:
        use_case = SearchImagesUseCase(
            settings, build_siglip_embedder(settings), SemanticIndexRepository(settings.semantic_index_dir)
        )
    except ValueError as exc:
        sys.exit(str(exc))
    if use_case.n_indexed == 0:
        sys.exit("nenhuma imagem indexada: rode antes `make index-images <pasta>`")

    outcome = use_case.search_text(args.query, args.k)
    response = outcome.response
    print(f'busca: "{response.query}" | {response.n_indexed} imagens indexadas')
    print(f"\n  {'#':>2}  {'imagem':<34} {'cosseno':>8}")
    for hit in response.hits:
        print(f"  {hit.rank:>2}  {hit.label:<34} {hit.score:8.3f}")

    args.out.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "_", response.query.lower()).strip("_")[:40] or "busca"
    path = write_output(args.out / f"search_{slug}.json", json.dumps(response.model_dump(), ensure_ascii=False, indent=2))
    print(f"\nresultado salvo em: {path}")
    if not args.no_plots:
        tiles = [(h.label, h.score, decode_image(p.read_bytes())) for h, p in zip(response.hits, outcome.paths)]
        sheet = write_output(args.out / f"search_{slug}.png", render_contact_sheet(response.query, tiles, response.n_indexed))
        print(f"folha de contato: {sheet}")


if __name__ == "__main__":
    main()
