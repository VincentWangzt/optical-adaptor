#!/usr/bin/env python3
from pathlib import Path

from optical_adaptor.infer_ocr import main

if __name__ == "__main__":
    config = Path(__file__).resolve().parents[1] / "configs/inference.deepseek-ocr.yaml"
    raise SystemExit(main(default_config=config))
