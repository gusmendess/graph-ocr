"""Ponto de entrada do `make run`.

    make run image6.jpeg                   # compara com a imagem base e gera TODAS as saídas
    make run original.png suspeita.png     # comparação explícita A x B (também gera todas as saídas)
    make run image6.jpeg BASE=outra.png    # troca a imagem base
    make run image6.jpeg ENGINE=tesseract  # troca o motor de OCR

Com uma imagem: cascata completa contra a base (`GRAPH_OCR_BASE_IMAGE`, padrão `image.png`) +
análise isolada da imagem suspeita. Saídas em outputs/.
"""

import argparse
import sys
from pathlib import Path

import analyze
import compare

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.api.settings import get_settings  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", type=Path, help="<suspeita> ou <referencia> <suspeita>")
    parser.add_argument("--base", type=Path, help="imagem base (padrão: GRAPH_OCR_BASE_IMAGE)")
    parser.add_argument("--engine", help="auto | tesseract | rapidocr | llm")
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if len(args.files) > 2:
        parser.error("informe 1 imagem (compara com a base) ou 2 (referência e suspeita)")
    if len(args.files) == 2:
        reference, suspect = args.files
    else:
        suspect = args.files[0]
        reference = args.base or get_settings().base_image

    for path in (reference, suspect):
        if not path.exists():
            parser.error(f"arquivo não encontrado: {path}")

    shared = ["--out", str(args.out)]
    if args.engine:
        shared += ["--engine", args.engine]
    if args.json:
        shared += ["--json"]

    print(f"== comparação: base={reference}  suspeita={suspect}")
    compare.main([str(reference), str(suspect), "--full", *shared])
    print(f"\n== análise isolada: {suspect}")
    analyze.main([str(suspect), *shared])

    stem_pair = f"{reference.stem}__vs__{suspect.stem}"
    print("\n== arquivos gerados")
    for name in (
        f"{stem_pair}.png",
        f"{stem_pair}__zoom.png",
        f"{stem_pair}__graphs.png",
        f"{suspect.stem}__analysis.png",
        f"{reference.stem}__ocr.txt",
        f"{suspect.stem}__ocr.txt",
    ):
        path = args.out / name
        print(f"  {'ok     ' if path.exists() else 'ausente'} {path}")


if __name__ == "__main__":
    main()
