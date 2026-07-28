#!/usr/bin/env python3
"""Check the optional PaddleOCR runtime without modifying the core environment."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from regdoc_ai.ocr.engines import EngineUnavailableError, PaddleOCRFieldEngine


def main() -> None:
    try:
        engine = PaddleOCRFieldEngine()
    except EngineUnavailableError as exc:
        print(json.dumps({"available": False, "reason": str(exc)}, indent=2))
        raise SystemExit(1)
    print(json.dumps({"available": True, "metadata": engine.metadata()}, indent=2))


if __name__ == "__main__":
    main()
