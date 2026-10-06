"""CLI: analisa UM comprovante (sem referência) com OCR + regras de consistência.

    uv run python scripts/analyze.py image5.jpeg
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.api.adapters.ocr.factory import build_engine  # noqa: E402
from app.api.infrastructure.output_io import write_output  # noqa: E402
from app.api.repositories.ocr_cache import OcrCacheRepository  # noqa: E402
from app.api.services.ocr_export_service import render_ocr_text  # noqa: E402
from app.api.services.ocr_service import OcrService  # noqa: E402
from app.api.services.visualization_service import render_document_report  # noqa: E402
from app.api.settings import get_settings  # noqa: E402
from app.api.use_cases.analyze_document import AnalyzeDocumentUseCase  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("doc", type=Path, help="comprovante a analisar")
    parser.add_argument("--engine", help="auto | tesseract | rapidocr | llm")
    parser.add_argument("--out", type=Path, default=Path("outputs"), help="pasta de saída")
    parser.add_argument("--json", action="store_true", help="imprime o relatório completo em JSON")
    args = parser.parse_args(argv)

    if not args.doc.exists():
        parser.error(f"arquivo não encontrado: {args.doc}")

    settings = get_settings()
    if args.engine:
        settings = settings.model_copy(update={"ocr_engine": args.engine})
    ocr = OcrService(build_engine(settings), OcrCacheRepository(settings.cache_dir))
    outcome = AnalyzeDocumentUseCase(settings, ocr).execute(args.doc.read_bytes())
    report = outcome.report

    print(f"veredito: {report.verdict.value}  ({report.n_words} palavras, OCR {report.ocr_engine})")
    print(f"resumo: {report.summary}")
    for flag in report.flags:
        words = ", ".join(repr(outcome.words[i].text) for i in flag.word_indexes)
        print(f"  [{flag.severity}] {flag.code}: {flag.detail}  (palavras: {words})")
    for entity in report.entities:
        print(f"  entidade ({entity.kind}): {entity.value}")
    if report.anomalies:
        weak = ", ".join(f"{a.text!r}={a.score:.2f}" for a in report.anomalies)
        print(f"  sinal geométrico fraco (não entra no veredito): {weak}")
    if args.json:
        print(report.model_dump_json(indent=2))

    args.out.mkdir(parents=True, exist_ok=True)
    target = write_output(args.out / f"{args.doc.stem}__analysis.png", render_document_report(outcome, args.doc.name))
    ocr_txt = write_output(args.out / f"{args.doc.stem}__ocr.txt", render_ocr_text(outcome.words, args.doc.name, report.ocr_engine))
    print(f"visualização: {target}")
    print(f"ocr (txt): {ocr_txt}")


if __name__ == "__main__":
    main()
