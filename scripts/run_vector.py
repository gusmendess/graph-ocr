"""Pipeline vetor + cosseno: score de 0 a 1 de quão similar a consulta é de cada documento indexado.

    make run-vector image7.png
    make run-vector image7.png K=8 WEIGHTS=0.2,0.6,0.2
    make run-vector image7.png INDEX=data/samples      # (re)constrói o índice antes de buscar

Só vetor + cosseno, sem pareamento por campo, sem grafo e sem veredito de adulteração:
  1. vetor multimodal da consulta: visual (DINOv2), texto (n-gramas do OCR) e layout (grade);
  2. cosseno de cada bloco contra o índice, truncado em [0, 1];
  3. score do par = média ponderada dos cossenos dos blocos (0 = nada parecido, 1 = idêntico).
O OCR roda só para alimentar os blocos de texto e de layout.
"""

import argparse
import json
import sys
from pathlib import Path

import index as index_script

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.api.adapters.ocr.factory import build_engine  # noqa: E402
from app.api.infrastructure.output_io import write_output  # noqa: E402
from app.api.repositories.index_repository import DocumentIndexRepository  # noqa: E402
from app.api.repositories.ocr_cache import OcrCacheRepository  # noqa: E402
from app.api.services.embedder_factory import build_document_embedder  # noqa: E402
from app.api.services.ocr_service import OcrService  # noqa: E402
from app.api.settings import get_settings  # noqa: E402
from app.api.use_cases.search_documents import SearchDocumentsUseCase  # noqa: E402


def parse_weights(raw: str | None, parser: argparse.ArgumentParser) -> dict[str, float] | None:
    if not raw:
        return None
    try:
        return dict(zip(("visual", "text", "layout"), (float(v) for v in raw.split(","))))
    except ValueError:
        parser.error("--weights espera três números separados por vírgula: visual,texto,layout")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("doc", type=Path, help="imagem consultada")
    parser.add_argument("-k", type=int, help="quantos pares (padrão: GRAPH_OCR_SEARCH_K)")
    parser.add_argument("--weights", help="pesos visual,texto,layout (ex.: 0.4,0.4,0.2)")
    parser.add_argument("--index", type=Path, help="pasta/imagens a indexar antes de buscar")
    parser.add_argument("--engine", help="auto | tesseract | rapidocr | llm")
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    if not args.doc.exists():
        parser.error(f"arquivo não encontrado: {args.doc}")
    weights = parse_weights(args.weights, parser)
    settings = get_settings()
    if args.engine:
        settings = settings.model_copy(update={"ocr_engine": args.engine})

    search = SearchDocumentsUseCase(
        settings,
        OcrService(build_engine(settings), OcrCacheRepository(settings.cache_dir)),
        build_document_embedder(settings),
        DocumentIndexRepository(settings.index_dir),
    )

    if args.index or search.n_indexed == 0:
        source = args.index or settings.index_source
        files = index_script.expand([source])
        if not files:
            sys.exit(f"nada para indexar em {source}")
        print(f"indexando {len(files)} arquivo(s) de {source}" + ("" if args.index else " (índice estava vazio)"))
        for file in files:
            search.add(file.read_bytes(), file.name, persist=False)
        search.save()

    response = search.search(args.doc.read_bytes(), args.k, weights).response
    blocks = response.blocks_used
    print(f"consulta: {args.doc.name} | indexados: {response.n_indexed} | blocos: {','.join(blocks)}")
    print("pesos: " + "  ".join(f"{b}={w:.2f}" for b, w in response.weights_used.items()))
    if response.exact_match:
        print("sha256 idêntico a um documento indexado (marcado com *)")

    print(f"\n  {'#':>2}  {'documento':<28} {'score 0-1':>9}  " + "  ".join(f"{b:>7}" for b in blocks))
    for n, hit in enumerate(response.hits, 1):
        scores = "  ".join(f"{hit.block_scores[b]:7.3f}" for b in blocks)
        print(f"{'*' if hit.is_query else ' '} {n:>2}  {hit.label:<28} {hit.score:9.3f}  {scores}")

    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "query": args.doc.name,
        "weights": response.weights_used,
        "blocks": blocks,
        "pairs": [
            {"document": h.label, "score": round(h.score, 6), "blocks": {b: round(h.block_scores[b], 6) for b in blocks},
             "same_file": h.is_query}
            for h in response.hits
        ],
    }
    path = write_output(args.out / f"{args.doc.stem}__similarity.json", json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nscores salvos em: {path}")


if __name__ == "__main__":
    main()
