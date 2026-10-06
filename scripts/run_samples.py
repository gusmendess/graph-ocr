"""Roda a cascata completa: original vs cada fraude e salva as visualizações em outputs/."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "data" / "samples"

for fraud in ("fraud_day.png", "fraud_day_time.png"):
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "compare.py"), str(SAMPLES / "original.png"), str(SAMPLES / fraud), "--full"],
        check=True,
    )
