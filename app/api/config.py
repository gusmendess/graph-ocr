import json
import logging
from pathlib import Path

APP_NAME = "graph-ocr"
VERSION_FILE = Path(__file__).resolve().parent.parent / "version.json"


def app_version() -> str:
    return json.loads(VERSION_FILE.read_text(encoding="utf-8"))["version"]


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
