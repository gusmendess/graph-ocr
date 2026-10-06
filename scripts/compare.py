"""CLI: compara dois comprovantes pela cascata e salva a visualização.

    uv run python scripts/compare.py data/samples/original.png data/samples/fraud_day.png --full
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.api.adapters.ocr.factory import build_engine  # noqa: E402
from app.api.repositories.ocr_cache import OcrCacheRepository  # noqa: E402
from app.api.services.ocr_export_service import render_ocr_text  # noqa: E402
from app.api.services.ocr_service import OcrService  # noqa: E402
from app.api.services.visualization_service import (  # noqa: E402
    render_comparison,
    render_evidence_zoom,
    render_graph_overlay,
)
from app.api.settings import get_settings  # noqa: E402
from app.api.use_cases.compare_documents import CompareDocumentsUseCase  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("doc_a", type=Path, help="documento de referência (original)")
    parser.add_argument("doc_b", type=Path, help="documento suspeito")
    parser.add_argument("--engine", help="auto | tesseract | rapidocr | llm")
    parser.add_argument("--out", type=Path, default=Path("outputs"), help="pasta de saída")
    parser.add_argument("--full", action="store_true", help="roda todas as camadas da cascata (mesmo após decidir)")
    parser.add_argument("--json", action="store_true", help="imprime o resultado completo em JSON")
    args = parser.parse_args(argv)

    settings = get_settings()
    if args.engine:
        settings = settings.model_copy(update={"ocr_engine": args.engine})
    ocr = OcrService(build_engine(settings), OcrCacheRepository(settings.cache_dir))
    use_case = CompareDocumentsUseCase(settings, ocr)

    outcome = use_case.execute(
        args.doc_a.read_bytes(), args.doc_b.read_bytes(), args.doc_a.stem, args.doc_b.stem, run_all=args.full
    )
    result = outcome.result

    print(f"veredito: {result.verdict.value}  score={result.score:.3f}  decidido por {result.decided_by}")
    for stage in result.stages:
        print(f"  [{stage.name:<16}] {stage.elapsed_ms:>8.1f} ms  score={stage.score:.3f}  {stage.detail}")
    print(f"resumo: {result.summary}")
    for shared in result.shared_entities:
        print(f"  entidade compartilhada ({shared.kind}): {shared.value} -> {', '.join(shared.documents)}")
    if args.json:
        print(result.model_dump_json(indent=2))

    args.out.mkdir(parents=True, exist_ok=True)
    name = f"{args.doc_a.stem}__vs__{args.doc_b.stem}"
    (args.out / f"{name}.png").write_bytes(render_comparison(outcome, args.doc_a.stem, args.doc_b.stem))
    zoom = render_evidence_zoom(outcome, args.doc_a.stem, args.doc_b.stem)
    if zoom:
        (args.out / f"{name}__zoom.png").write_bytes(zoom)
    if outcome.words_a is not None:  # o OCR só roda se a cascata chegou na camada 3
        for doc, words in ((args.doc_a, outcome.words_a), (args.doc_b, outcome.words_b)):
            path = args.out / f"{doc.stem}__ocr.txt"
            path.write_text(render_ocr_text(words, doc.name, result.ocr_engine), encoding="utf-8")
            print(f"ocr (txt): {path}")
    overlay = render_graph_overlay(outcome, args.doc_a.stem, args.doc_b.stem)
    if overlay:
        (args.out / f"{name}__graphs.png").write_bytes(overlay)
    print(f"visualização: {args.out / (name + '.png')}")


if __name__ == "__main__":
    main()
