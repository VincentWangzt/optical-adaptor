from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import polars as pl
import torch
from torch.utils.data import Dataset, Sampler

from optical_adaptor.automodel.config import (
    DataConfig,
    FilterConfig,
    OpticalConfig,
    Source,
    fingerprint,
    preparation_fingerprint,
)
from optical_adaptor.automodel.processing import ConversationCompiler, RejectedSample


def validate_preparation(optical: OpticalConfig, seed: int) -> None:
    root = Path(optical.prepare.output_dir)
    if (root / "preparation.incomplete").exists():
        raise ValueError("Data preparation is incomplete")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary["preparation_fingerprint"] != preparation_fingerprint(optical, seed):
        raise ValueError(
            "Prepared data does not match the requested source limits, revisions, prompts, "
            "pagination, seed, or renderer. Prepare a fresh DATA_DIR."
        )


class ConversationDataset(Dataset):
    def __init__(self, root: str, split: str, filters: FilterConfig, per_slice: int | None):
        self.root = Path(root)
        if (self.root / "preparation.incomplete").exists():
            raise ValueError("Data preparation is incomplete")
        frame = pl.read_parquet(self.root / "manifest.parquet").filter(pl.col("split") == split)
        for key, column in (
            ("sources", "source"),
            ("tasks", "task"),
            ("views", "view_family"),
            ("image_bins", "image_bin"),
            ("turn_bins", "turn_bin"),
        ):
            values = getattr(filters, key)
            if values is not None:
                if column == "view_family":
                    frame = frame.with_columns(
                        pl.col("slice").str.split("/").list.get(2).alias(column)
                    )
                frame = frame.filter(pl.col(column).is_in(values))
        frame = frame.filter(pl.col("image_count") >= filters.min_images)
        if filters.max_images is not None:
            frame = frame.filter(pl.col("image_count") <= filters.max_images)
        frame = frame.sort("sample_id")
        if per_slice is not None:
            frame = frame.group_by("slice", maintain_order=True).head(per_slice)
        self.rows = frame.to_dicts()
        if not self.rows:
            raise ValueError(f"No {split} examples match the requested data filters")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with (self.root / row["path"]).open("rb") as handle:
            handle.seek(row["offset"])
            record = json.loads(handle.read(row["length"]))
        if fingerprint(record) != row["sha256"]:
            raise ValueError(f"Data integrity failure for {row['sample_id']}")
        return record

    def preflight(self, compiler: ConversationCompiler) -> tuple[list[int], dict]:
        kept, dropped, lengths = [], Counter(), {}
        for index, row in enumerate(self.rows):
            try:
                paired = compiler.compile(self[index])
            except RejectedSample as error:
                if compiler.config.overlength == "error":
                    raise ValueError(f"Rejected {row['sample_id']}: {error}") from error
                dropped[f"{row['slice']}/{error}"] += 1
                continue
            kept.append(index)
            lengths[row["sample_id"]] = {
                "teacher": len(paired.teacher_ids),
                "student": len(paired.student_ids),
                "targets": len(paired.targets),
            }
        if not kept:
            raise ValueError(f"No examples survive preflight: {dict(dropped)}")
        return kept, {"kept": len(kept), "dropped": dict(dropped), "lengths": lengths}

    def select(self, indices: list[int]):
        self.rows = [self.rows[index] for index in indices]


class MixtureSampler(Sampler[int]):
    """Deterministic global draws, divided across DP ranks; all ranks take equal steps."""

    def __init__(
        self,
        dataset: ConversationDataset,
        sources: list[Source],
        config: DataConfig,
        seed: int,
        rank: int,
        world_size: int,
    ):
        source_weights = {source.name: source.weight for source in sources}
        counts = Counter(row["slice"] for row in dataset.rows)
        source_tasks = Counter((row["source"], row["task"]) for row in dataset.rows)
        slices_per_task = Counter((key.split("/")[0], key.split("/")[1]) for key in counts)
        task_mass = Counter()
        for source, task in source_tasks:
            task_mass[source] += config.task_weights[task]
        weights = []
        for row in dataset.rows:
            weight = (
                source_weights[row["source"]]
                * config.task_weights[row["task"]]
                / task_mass[row["source"]]
            )
            if config.balance_slices:
                weight /= counts[row["slice"]] * slices_per_task[row["source"], row["task"]]
            else:
                weight /= source_tasks[row["source"], row["task"]]
            weights.append(weight)
        self.weights = torch.tensor(weights, dtype=torch.double)
        if not torch.isfinite(self.weights).all() or (self.weights <= 0).any():
            raise ValueError("Mixture weights must be positive and finite")
        requested = config.samples_per_epoch or len(dataset)
        self.num_samples = math.ceil(requested / world_size)
        self.rank, self.world_size, self.seed = rank, world_size, seed
        self.epoch, self.cursor = 0, 0

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        global_indices = torch.multinomial(
            self.weights, self.num_samples * self.world_size, replacement=True, generator=generator
        ).tolist()
        indices = global_indices[self.rank :: self.world_size]
        while self.cursor < len(indices):
            index = indices[self.cursor]
            self.cursor += 1
            yield index

    def set_epoch(self, epoch):
        if epoch != self.epoch:
            self.epoch, self.cursor = epoch, 0

    def state_dict(self):
        return {"epoch": self.epoch, "cursor": self.cursor}

    def load_state_dict(self, state):
        self.epoch, self.cursor = state["epoch"], state["cursor"]
