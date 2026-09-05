from __future__ import annotations

import argparse
import json
import os
from collections import OrderedDict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from optical_adaptor.renderer import render_pages
from optical_adaptor.training.config import (
    Pipeline,
    file_sha256,
    fingerprint,
    load_credentials,
    load_pipeline,
    write_json,
)
from optical_adaptor.training.data import load_manifest
from optical_adaptor.training.models import DeepSeekVision, FrozenQwen


def shard_records(records: list[dict], kind: str) -> dict[int, list[dict]]:
    groups: dict[int, list[dict]] = {}
    for record in records:
        shard = record[f"{kind}_shard"]
        if shard >= 0:
            groups.setdefault(shard, []).append(record)
    for rows in groups.values():
        if [row[f"{kind}_row"] for row in rows] != list(range(len(rows))):
            raise ValueError("cache row indices are not contiguous")
    return groups


def shard_path(pipeline: Pipeline, kind: str, index: int) -> Path:
    return pipeline.output / kind / f"shard-{index:05d}.safetensors"


def shard_fingerprint(pipeline: Pipeline, kind: str, records: list[dict]) -> str:
    return fingerprint(
        {
            "schema": 1,
            "kind": kind,
            "data": pipeline.data_fingerprint,
            "records": [row["record_hash"] for row in records],
        }
    )


def verify_shard(pipeline: Pipeline, kind: str, index: int, records: list[dict]) -> bool:
    path = shard_path(pipeline, kind, index)
    metadata_path = path.with_suffix(".json")
    if not metadata_path.exists():
        return False
    metadata = json.loads(metadata_path.read_text())
    if metadata["fingerprint"] != shard_fingerprint(pipeline, kind, records):
        raise ValueError(f"cache fingerprint mismatch: {path}")
    if not path.exists() or file_sha256(path) != metadata["sha256"]:
        raise ValueError(f"cache checksum mismatch: {path}")
    shape = (
        [len(records), 111, 1280]
        if kind == "encoder"
        else [len(records), pipeline.config.data.continuation_tokens, 2560]
    )
    with safe_open(path, framework="pt") as stream:
        if list(stream.keys()) != ["values"]:
            raise ValueError(f"unexpected tensor keys: {path}")
        tensor = stream.get_slice("values")
        if tensor.get_shape() != shape or tensor.get_dtype() != "BF16":
            raise ValueError(f"cache shape/dtype mismatch: {path}")
    return True


def build_cache(pipeline: Pipeline, kind: str, batch_size: int) -> None:
    if kind not in {"encoder", "teacher"} or batch_size < 1:
        raise ValueError("invalid cache kind or batch size")
    if not os.environ.get("CUDA_VISIBLE_DEVICES") or not torch.cuda.is_available():
        raise RuntimeError("select idle GPUs explicitly with CUDA_VISIBLE_DEVICES")
    load_credentials(pipeline, wandb=False)
    records = load_manifest(pipeline)
    groups = shard_records(records, kind)
    pending = [
        index for index, rows in groups.items() if not verify_shard(pipeline, kind, index, rows)
    ]
    if not pending:
        print(f"all {kind} shards verified", flush=True)
        return
    device = torch.device("cuda:0")
    model = DeepSeekVision(pipeline, device) if kind == "encoder" else FrozenQwen(pipeline, device)
    for index in pending:
        rows, values = groups[index], []
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            if kind == "encoder":
                images = [
                    render_pages(row["visual"], config=pipeline.render)[0][0] for row in batch
                ]
                output = model(images)
            else:
                output = model.teacher_hidden(batch)
            if not torch.isfinite(output).all():
                raise RuntimeError("non-finite cache values")
            values.append(output.detach().to(device="cpu", dtype=torch.bfloat16))
        path = shard_path(pipeline, kind, index)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        save_file(
            {"values": torch.cat(values).contiguous()},
            str(temporary),
            metadata={"fingerprint": shard_fingerprint(pipeline, kind, rows)},
        )
        temporary.replace(path)
        write_json(
            path.with_suffix(".json"),
            {"fingerprint": shard_fingerprint(pipeline, kind, rows), "sha256": file_sha256(path)},
        )
        verify_shard(pipeline, kind, index, rows)
        print(f"{kind} shard {index + 1}/{len(groups)} complete", flush=True)


class TensorCache:
    def __init__(self, pipeline: Pipeline, records: list[dict], *, max_open: int = 4):
        self.pipeline, self.max_open = pipeline, max_open
        self.loaded: OrderedDict[tuple[str, int], torch.Tensor] = OrderedDict()
        for kind in ("encoder", "teacher"):
            for index, rows in shard_records(records, kind).items():
                if not verify_shard(pipeline, kind, index, rows):
                    raise ValueError(f"missing {kind} cache shard {index}")

    def batch(self, records: list[dict], kind: str, device: torch.device) -> torch.Tensor:
        values = []
        for row in records:
            key = kind, row[f"{kind}_shard"]
            if key not in self.loaded:
                self.loaded[key] = load_file(str(shard_path(self.pipeline, *key)))["values"]
            self.loaded.move_to_end(key)
            values.append(self.loaded[key][row[f"{kind}_row"]])
            while len(self.loaded) > self.max_open:
                self.loaded.popitem(last=False)
        return torch.stack(values).to(device=device, non_blocking=True)


def cache_main(kind: str) -> None:
    parser = argparse.ArgumentParser(description=f"Build verified frozen {kind} cache")
    parser.add_argument("--config", type=Path, default=Path("configs/training.yaml"))
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    build_cache(load_pipeline(args.config), kind, args.batch_size)


def extract_main() -> None:
    cache_main("encoder")


def teacher_main() -> None:
    cache_main("teacher")
