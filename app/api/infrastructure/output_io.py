import sys
import time
from datetime import datetime
from pathlib import Path


def write_output(path: Path, data: bytes | str, retries: int = 3, delay: float = 0.3) -> Path:
    """Grava uma saída e devolve o caminho realmente usado.

    No Windows, um arquivo aberto em outro programa (ex.: pré-visualização da imagem no editor) pode
    recusar a escrita (OSError 22/13). Tenta de novo algumas vezes; se persistir, grava ao lado com
    sufixo de hora e avisa, em vez de derrubar o `make run` depois de a análise já ter terminado.
    """
    payload = data.encode("utf-8") if isinstance(data, str) else data
    error: OSError | None = None
    for _ in range(retries):
        try:
            path.write_bytes(payload)
            return path
        except OSError as exc:
            error = exc
            time.sleep(delay)
    alternative = path.with_name(f"{path.stem}__{datetime.now():%H%M%S}{path.suffix}")
    alternative.write_bytes(payload)
    print(
        f"aviso: não consegui gravar {path.name} ({error}); salvei como {alternative.name}. "
        "O arquivo provavelmente está aberto em outro programa.",
        file=sys.stderr,
    )
    return alternative
