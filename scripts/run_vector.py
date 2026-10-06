"""Pipeline de vetor: busca reversa multimodal -> OCR + grafo nos candidatos -> saídas da melhor referência.

    make run-vector image7.png
    make run-vector image7.png K=8 TOP=2 WEIGHTS=0.2,0.6,0.2
    make run-vector image7.png INDEX=data/samples      # (re)constrói o índice antes de buscar

Etapas:
  1. sha256 + vetor multimodal (visual, texto, layout) da imagem consultada;
  2. top-k por cosseno no índice;
  3. cascata de adulteração (OCR + grafo) em cada candidato;
  4. a referência é o candidato que melhor explica a consulta (maior fração de conteúdo casado);
     para ela saem todas as imagens e textos, como no `make run`.
"""

import argparse
import sys
import tempfile
from pathlib import Path

import analyze
import compare
import index as index_script

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.api.adapters.ocr.factory import build_engine  # noqa: E402
from app.api.domain.enums import WordStatus  # noqa: E402
from app.api.repositories.index_repository import DocumentIndexRepository  # noqa: E402
from app.api.repositories.ocr_cache import OcrCacheRepository  # noqa: E402
from app.api.services.embedder_factory import build_document_embedder  # noqa: E402
from app.api.services.ocr_service import OcrService  # noqa: E402
from app.api.settings import get_settings  # noqa: E402
from app.api.use_cases.compare_documents import CompareDocumentsUseCase  # noqa: E402
from app.api.use_cases.search_documents import SearchDocumentsUseCase, rank_references  # noqa: E402

_EDITS = (WordStatus.MODIFIED, WordStatus.ADDED, WordStatus.REMOVED)


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
    parser.add_argument("-k", type=int, help="candidatos da busca (padrão: GRAPH_OCR_SEARCH_K)")
    parser.add_argument("--top", type=int, default=1, help="quantas referências geram saídas completas (padrão: 1)")
    parser.add_argument("--weights", help="pesos visual,texto,layout (ex.: 0.4,0.4,0.2)")
    parser.add_argument("--index", type=Path, help="pasta/imagens a indexar antes de buscar (reconstrói o índice)")
    parser.add_argument("--engine", help="auto | tesseract | rapidocr | llm")
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    if not args.doc.exists():
        parser.error(f"arquivo não encontrado: {args.doc}")
    weights = parse_weights(args.weights, parser)
    settings = get_settings()
    if args.engine:
        settings = settings.model_copy(update={"ocr_engine": args.engine})

    ocr = OcrService(build_engine(settings), OcrCacheRepository(settings.cache_dir))
    search = SearchDocumentsUseCase(
        settings, ocr, build_document_embedder(settings), DocumentIndexRepository(settings.index_dir)
    )

    print("== índice")
    if args.index or search.n_indexed == 0:
        source = args.index or settings.index_source
        files = index_script.expand([source])
        if not files:
            sys.exit(f"nada para indexar em {source}")
        print(f"indexando {len(files)} arquivo(s) de {source}" + ("" if args.index else " (índice estava vazio)"))
        for file in files:
            search.add(file.read_bytes(), file.name, persist=False)
        search.save()
    print(f"{search.n_indexed} documento(s) indexado(s) em {settings.index_dir}")

    data = args.doc.read_bytes()
    compare_use_case = CompareDocumentsUseCase(settings, ocr)
    rows = search.search_and_compare(data, compare_use_case, args.k, weights)
    response = search.search(data, args.k, weights).response

    print(f"\n== busca vetorial: {args.doc.name} | blocos: {','.join(response.blocks_used)} | "
          + "  ".join(f"{b}={w:.2f}" for b, w in response.weights_used.items()))
    if response.exact_match:
        print("sha256 idêntico a um documento indexado (marcado com *)")
    cols = response.blocks_used
    print(f"\n  {'#':>2}  {'documento':<26} {'composto':>8}  " + "  ".join(f"{c:>6}" for c in cols)
          + f"  {'conteúdo':>8}  {'edições':>7}  veredito")
    for n, (hit, result) in enumerate(rows, 1):
        blocks = "  ".join(f"{hit.block_scores[c]:6.3f}" for c in cols)
        if result is None:
            tail = f"  {'':>8}  {'':>7}  (a própria consulta)"
        else:
            edits = sum(1 for m in result.matches if m.status in _EDITS)
            tail = f"  {result.score:8.3f}  {edits:7d}  {result.verdict.value}"
        print(f"{'*' if hit.is_query else ' '} {n:>2}  {hit.label:<26} {hit.score:8.3f}  {blocks}{tail}")

    ranked = rank_references(rows)
    if not ranked:
        sys.exit("nenhum candidato diferente da própria consulta no índice")
    print("\n== referências (melhor explica a consulta: maior conteúdo casado; desempate pelo cosseno)")
    for n, (hit, result) in enumerate(ranked[: args.top], 1):
        print(f"  {n}. {hit.label}  conteúdo={result.score:.3f}  cosseno={hit.score:.3f}  -> {result.verdict.value}")

    shared = ["--out", str(args.out)] + (["--engine", args.engine] if args.engine else [])
    generated: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        for hit, _ in ranked[: args.top]:
            reference = Path(tmp) / hit.label  # com o nome original, para as saídas saírem com ele
            reference.write_bytes(search.document_path(hit.doc_id).read_bytes())
            print(f"\n== comparação: referência={hit.label}  suspeita={args.doc.name}")
            compare.main([str(reference), str(args.doc), "--full", *shared])
            pair = f"{reference.stem}__vs__{args.doc.stem}"
            generated += [f"{pair}.png", f"{pair}__zoom.png", f"{pair}__graphs.png", f"{reference.stem}__ocr.txt"]

    print(f"\n== análise isolada: {args.doc.name}")
    analyze.main([str(args.doc), *shared])
    generated += [f"{args.doc.stem}__analysis.png", f"{args.doc.stem}__ocr.txt"]

    print("\n== arquivos gerados")
    for name in dict.fromkeys(generated):
        path = args.out / name
        print(f"  {'ok     ' if path.exists() else 'ausente'} {path}")


if __name__ == "__main__":
    main()
