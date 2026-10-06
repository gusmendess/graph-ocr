from pathlib import Path

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp"})


def expand_images(paths: list[Path]) -> list[Path]:
    """Arquivos de imagem a partir de uma lista de arquivos e/ou pastas (pastas não são percorridas recursivamente).

    Os arquivos vêm na ordem em que foram passados; dentro de uma pasta, em ordem alfabética. Um caminho que não
    existe vira FileNotFoundError, em vez de ser ignorado em silêncio.
    """
    files: list[Path] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"não encontrado: {path}")
        if path.is_dir():
            files += sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        else:
            files.append(path)
    return files
