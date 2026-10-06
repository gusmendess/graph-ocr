"""Cascata sha256 -> phash -> vetor: score de 0 a 1 de quão similar a consulta é de cada documento indexado.

    make run-vector <imagem>
    make run-vector <imagem> K=8 WEIGHTS=0.05,0.9,0.05
    make run-vector <imagem> INDEX=<pasta>             # (re)constrói o índice antes de buscar

Cada par (consulta, documento indexado) para no primeiro estágio que decide:
  1. sha256: bytes idênticos -> score 1.0;
  2. phash:  similaridade perceptual abaixo do limiar -> o phash decide (score = a similaridade), sem OCR;
  3. vetor:  só para o que sobrou: cosseno do vetor multimodal (visual + texto + layout), de 0 a 1.
O OCR e a rede neural só rodam se algum par chegar ao estágio 3.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.api.adapters.ocr.factory import build_engine  # noqa: E402
from app.api.infrastructure.image_files import expand_images  # noqa: E402
from app.api.infrastructure.image_io import decode_image  # noqa: E402
from app.api.infrastructure.output_io import write_output  # noqa: E402
from app.api.repositories.index_repository import DocumentIndexRepository  # noqa: E402
from app.api.repositories.ocr_cache import OcrCacheRepository  # noqa: E402
from app.api.services.embedder_factory import build_document_embedder  # noqa: E402
from app.api.services.hash_visualization_service import HashCandidate, render_pair_figures  # noqa: E402
from app.api.services.ocr_export_service import render_ocr_text  # noqa: E402
from app.api.services.ocr_service import OcrService  # noqa: E402
from app.api.services.vector_visualization_service import build_comparison, render_vector_panels  # noqa: E402
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
    parser.add_argument("--weights", help="pesos visual,texto,layout (ex.: 0.05,0.9,0.05)")
    parser.add_argument("--index", type=Path, help="pasta/imagens a indexar antes de buscar")
    parser.add_argument("--engine", help="auto | tesseract | rapidocr | llm")
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    parser.add_argument("--no-plots", action="store_true", help="não gera as figuras para slides")
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
        try:
            files = expand_images([source])
        except FileNotFoundError as exc:
            sys.exit(str(exc))
        if not files:
            sys.exit(f"nada para indexar em {source}")
        print(f"indexando {len(files)} arquivo(s) de {source}" + ("" if args.index else " (índice estava vazio)"))
        for file in files:
            search.add(file.read_bytes(), file.name, persist=False)
        search.save()

    outcome = search.search(args.doc.read_bytes(), args.k, weights)
    response = outcome.response
    blocks = response.blocks_used
    stages = Counter(response.stage_counts)  # entre TODOS os indexados, não só os k exibidos
    print(f"consulta: {args.doc.name} | indexados: {response.n_indexed}")
    print(f"cascata: sha256 -> phash (limiar {response.phash_threshold}) -> vetor | "
          f"decididos aqui: sha256={stages['sha256']}, phash={stages['phash']}, vetor={stages['vector']}")
    if response.vector_computed:
        print("pesos: " + "  ".join(f"{b}={w:.2f}" for b, w in response.weights_used.items()))

    print(f"\n  {'#':>2}  {'documento':<26} {'score 0-1':>9}  {'decidido por':<12} {'phash':>6}  "
          + "  ".join(f"{b:>7}" for b in blocks))
    for n, hit in enumerate(response.hits, 1):
        phash = f"{hit.phash:6.3f}" if hit.phash is not None else f"{'-':>6}"
        scores = "  ".join(f"{hit.block_scores[b]:7.3f}" if b in hit.block_scores else f"{'-':>7}" for b in blocks)
        print(f"{'*' if hit.is_query else ' '} {n:>2}  {hit.label:<26} {hit.score:9.3f}  {hit.decided_by:<12} {phash}  {scores}")
    if stages["phash"]:
        print(f"\n{stages['phash']} par(es) decidido(s) pelo phash (< {response.phash_threshold}) não passaram pelo vetor. "
              "Para forçar o vetor em todos: GRAPH_OCR_PHASH_THRESHOLD=0")

    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "query": args.doc.name,
        "phash_threshold": response.phash_threshold,
        "vector_computed": response.vector_computed,
        "weights": response.weights_used,
        "pairs": [
            {"document": h.label, "score": round(h.score, 6), "decided_by": h.decided_by,
             "phash": None if h.phash is None else round(h.phash, 6),
             "blocks": {b: round(v, 6) for b, v in h.block_scores.items()}, "same_file": h.is_query}
            for h in response.hits
        ],
    }
    path = write_output(args.out / f"{args.doc.stem}__similarity.json", json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nscores salvos em: {path}")

    if not args.no_plots and response.hits:  # sha256 + phash: uma figura por par e por algoritmo; não dependem de OCR nem de vetor
        query_image = decode_image(args.doc.read_bytes())
        stems = [Path(h.label).stem for h in response.hits]
        print("figuras por par e algoritmo (sha256 e phash):")
        for hit, stem in zip(response.hits, stems):
            if stems.count(stem) > 1:  # mesmo nome base com extensões diferentes: mantém a extensão para não sobrescrever
                stem = Path(hit.label).name.replace(".", "_")
            candidate = HashCandidate(hit.label, hit.doc_id, decode_image(search.document_path(hit.doc_id).read_bytes()))
            figures = render_pair_figures(args.doc.name, outcome.sha, query_image, candidate, response.phash_threshold)
            for algorithm, png in figures.items():
                print(f"  {write_output(args.out / f'{args.doc.stem}_vs_{stem}_{algorithm}.png', png)}")

    if outcome.words is not None:  # o OCR só roda se algum par chegou ao estágio de vetor
        ocr_path = write_output(
            args.out / f"{args.doc.stem}__ocr.txt", render_ocr_text(outcome.words, args.doc.name, search.engine_name)
        )
        print(f"ocr (txt): {ocr_path}")
    else:
        print("ocr (txt) e figuras de vetores: não gerados, porque nenhum par chegou ao estágio de vetor (OCR não rodou)")

    if outcome.query is not None and not args.no_plots:
        comparison = build_comparison(
            args.doc.name, outcome.sha, outcome.query, search.index, weights or settings.search_weights()
        )
        print("figuras para os slides:")
        for name, png in render_vector_panels(comparison).items():
            print(f"  {write_output(args.out / f'{args.doc.stem}__vectors_{name}.png', png)}")


if __name__ == "__main__":
    main()
