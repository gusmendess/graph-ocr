"""CLI: busca reversa por vetor multimodal (cosseno) e, opcionalmente, cascata de adulteração nos candidatos.

    uv run python scripts/search.py image6.jpeg -k 5
    uv run python scripts/search.py image6.jpeg --compare            # OCR + grafo em cada candidato
    uv run python scripts/search.py image6.jpeg --weights 0.2,0.6,0.2  # visual,texto,layout
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
from app.api.use_cases.compare_documents import CompareDocumentsUseCase  # noqa: E402
from app.api.use_cases.search_documents import SearchDocumentsUseCase  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("doc", type=Path, help="documento consultado")
    parser.add_argument("-k", type=int, help="quantos candidatos (padrão: GRAPH_OCR_SEARCH_K)")
    parser.add_argument("--weights", help="pesos visual,texto,layout (ex.: 0.4,0.4,0.2)")
    parser.add_argument("--compare", action="store_true", help="roda OCR + grafo em cada candidato")
    parser.add_argument("--engine", help="auto | tesseract | rapidocr | llm")
    args = parser.parse_args(argv)

    if not args.doc.exists():
        parser.error(f"arquivo não encontrado: {args.doc}")
    settings = get_settings()
    if args.engine:
        settings = settings.model_copy(update={"ocr_engine": args.engine})
    weights = None
    if args.weights:
        try:
            weights = dict(zip(("visual", "text", "layout"), (float(v) for v in args.weights.split(","))))
        except ValueError:
            parser.error("--weights espera três números separados por vírgula: visual,texto,layout")

    ocr = OcrService(build_engine(settings), OcrCacheRepository(settings.cache_dir))
    use_case = SearchDocumentsUseCase(
        settings, ocr, build_document_embedder(settings), DocumentIndexRepository(settings.index_dir)
    )
    if use_case.n_indexed == 0:
        sys.exit("índice vazio: rode antes `make index <pasta ou imagens>`")

    data = args.doc.read_bytes()
    if args.compare:
        rows = use_case.search_and_compare(data, CompareDocumentsUseCase(settings, ocr), args.k, weights)
        response = use_case.search(data, args.k, weights).response
    else:
        response = use_case.search(data, args.k, weights).response
        rows = [(hit, None) for hit in response.hits]

    print(f"consulta: {args.doc.name} | indexados: {response.n_indexed} | blocos: {','.join(response.blocks_used)}")
    print("pesos: " + "  ".join(f"{b}={w:.2f}" for b, w in response.weights_used.items()))
    if response.exact_match:
        print("sha256 idêntico a um documento indexado (marcado com *)")
    cols = response.blocks_used
    print(f"\n  {'#':>2}  {'documento':<28} {'composto':>8}  " + "  ".join(f"{c:>7}" for c in cols) + ("  veredito" if args.compare else ""))
    for n, (hit, result) in enumerate(rows, 1):
        mark = "*" if hit.is_query else " "
        scores = "  ".join(f"{hit.block_scores[c]:7.3f}" for c in cols)
        verdict = ""
        if args.compare:
            verdict = "  (a própria consulta)" if result is None else f"  {result.verdict.value}"
        print(f"{mark} {n:>2}  {hit.label:<28} {hit.score:8.3f}  {scores}{verdict}")
        if result is not None and result.summary:
            print(f"        {result.summary[:150]}")


if __name__ == "__main__":
    main()
