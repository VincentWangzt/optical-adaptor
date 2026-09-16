"""Shared artifact I/O and the canonical optical configuration context."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from pydantic import Field

from optical_adaptor.automodel.config import OpticalConfig, fingerprint, read_config
from optical_adaptor.renderer import RenderConfig, load_render_config

PositiveInt = Annotated[int, Field(gt=0)]
Revision = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True)
class Pipeline:
    optical: OpticalConfig
    repo: Path
    render: RenderConfig
    data_fingerprint: str

    @property
    def manifest(self):
        return self.repo / self.optical.prepare.output_dir / "manifest.parquet"


def load_pipeline(path: str | Path) -> Pipeline:
    path = Path(path).resolve()
    _, optical = read_config(path)
    repo = path.parent.parent
    render_path = repo / optical.render_config
    render = load_render_config(render_path)
    identity = fingerprint(
        {
            "optical": optical.model_dump(),
            "render": file_sha256(render_path),
            "fonts": [
                file_sha256(Path(p)) for p in (render.text.font, *render.text.fallback_fonts)
            ],
        }
    )
    return Pipeline(optical, repo, render, identity)


def load_credentials(pipeline: Pipeline, *, wandb: bool) -> None:
    load_dotenv(pipeline.repo / ".env", override=False)
    if wandb and not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("Online W&B logging requires WANDB_API_KEY")
