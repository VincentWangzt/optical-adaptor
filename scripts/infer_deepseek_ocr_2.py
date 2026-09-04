#!/usr/bin/env python3
from pathlib import Path

from optical_agent.infer_ocr import main

if __name__ == "__main__":
    config = Path(__file__).resolve().parents[1] / "configs/inference.deepseek-ocr-2.yaml"
    raise SystemExit(main(default_config=config))
