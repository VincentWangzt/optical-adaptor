from __future__ import annotations

import json
import logging
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
    fingerprint,
    preparation_fingerprint,
)
from optical_adaptor.automodel.processing import ConversationCompiler, RejectedSample


def inherited_values(values: dict, leaves: list[str], *, integer: bool = False) -> dict:
    """Nearest ancestor replaces the default; ancestors do not partition mass."""
    valid = {""}
    for leaf in leaves:
        parts = leaf.split("/")
        valid.update("/".join(parts[:i]) for i in range(1, len(parts) + 1))
    if "" not in values:
        raise ValueError("Hierarchy must configure the root default under the empty key")
    unknown = values.keys() - valid
    if unknown:
        raise ValueError(f"Unknown hierarchy paths: {sorted(unknown)}")
    for key, value in values.items():
        if not math.isfinite(value) or value < 0 or (integer and type(value) is not int):
            raise ValueError(f"Invalid {'count' if integer else 'weight'} at {key!r}: {value}")
    resolved = {}
    for leaf in leaves:
        parts = leaf.split("/")
        ancestors = ["/".join(parts[:i]) for i in range(len(parts), -1, -1)]
        resolved[leaf] = next(values[path] for path in ancestors if path in values)
    return resolved


def removal_report(before: list[dict], after: list[dict], stage: str) -> dict:
    original, retained = Counter(r["slice"] for r in before), Counter(r["slice"] for r in after)
    report = {}
    for key, count in sorted(original.items()):
        removed = count - retained[key]
        report[key] = {
            "before": count,
            "retained": retained[key],
            "removed": removed,
            "removed_percent": 100 * removed / count,
        }
        if removed:
            logging.warning(
                "%s %s: %d before, %d retained, %d removed (%.2f%% of stage input)",
                stage,
                key,
                count,
                retained[key],
                removed,
                100 * removed / count,
            )
    return report


def select_evaluation(
    rows: list[dict], requested: dict[str, int], seed: int
) -> tuple[list[int], dict]:
    counts, chosen, report = Counter(), [], {}
    for index in sorted(range(len(rows)), key=lambda i: fingerprint([seed, rows[i]["sample_id"]])):
        key = rows[index]["slice"]
        if counts[key] < requested[key]:
            chosen.append(index)
            counts[key] += 1
    for key, wanted in requested.items():
        shortfall = wanted - counts[key]
        report[key] = {
            "requested": wanted,
            "selected": counts[key],
            "shortfall": shortfall,
            "shortfall_percent": 100 * shortfall / wanted if wanted else 0,
        }
        if shortfall:
            logging.warning(
                "Evaluation %s: %d requested, %d available, shortfall %d (%.2f%%)",
                key,
                wanted,
                counts[key],
                shortfall,
                100 * shortfall / wanted,
            )
    return sorted(chosen), report


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
    def __init__(self, root: str, split: str, filters: FilterConfig):
        self.root = Path(root)
        if (self.root / "preparation.incomplete").exists():
            raise ValueError("Data preparation is incomplete")
        manifest = pl.read_parquet(self.root / "manifest.parquet")
        self.all_leaves = sorted(manifest["slice"].unique().to_list())
        frame = manifest.filter(pl.col("split") == split)
        before = frame.to_dicts()
        for key, column in (
            ("sources", "source"),
            ("tasks", "task"),
            ("image_bins", "image_bin"),
            ("turn_bins", "turn_bin"),
        ):
            values = getattr(filters, key)
            if values is not None:
                unknown = set(values) - set(manifest[column].unique().to_list())
                if unknown:
                    raise ValueError(f"Unknown {key}: {sorted(unknown)}")
                frame = frame.filter(pl.col(column).is_in(values))
        self.selected_leaves = sorted(
            key
            for key in self.all_leaves
            if (filters.sources is None or key.split("/")[0] in filters.sources)
            and (filters.tasks is None or key.split("/")[1] in filters.tasks)
            and (
                filters.image_bins is None
                or not key.split("/")[2].startswith("images-")
                or key.split("/")[2].removeprefix("images-") in filters.image_bins
            )
            and (
                filters.turn_bins is None
                or not key.split("/")[2].startswith("turns-")
                or key.split("/")[2].removeprefix("turns-") in filters.turn_bins
            )
        )
        frame = frame.filter(pl.col("image_count") >= filters.min_images)
        if filters.max_images is not None:
            frame = frame.filter(pl.col("image_count") <= filters.max_images)
        frame = frame.sort("sample_id")
        self.rows = frame.to_dicts()
        self.filter_report = removal_report(before, self.rows, f"{split} filters")
        if not self.rows and split == "train":
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

    def preflight(self, compiler: ConversationCompiler, indices: range) -> tuple[list[int], dict]:
        kept, dropped, lengths = [], Counter(), {}
        for processed, index in enumerate(indices, start=1):
            row = self.rows[index]
            if processed % 1000 == 0:
                logging.info("Preflight: %d/%d assigned records", processed, len(indices))
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
        return kept, {"kept": len(kept), "dropped": dict(dropped), "lengths": lengths}

    def select(self, indices: list[int]):
        self.rows = [self.rows[index] for index in indices]


class MixtureSampler(Sampler[int]):
    """Deterministic global draws, divided across DP ranks; all ranks take equal steps."""

    def __init__(
        self,
        dataset: ConversationDataset,
        config: DataConfig,
        seed: int,
        rank: int,
        world_size: int,
        global_batch_size: int,
    ):
        counts = Counter(row["slice"] for row in dataset.rows)
        resolved = inherited_values(config.weights, dataset.all_leaves)
        total = sum(resolved[key] for key in counts)
        if total <= 0:
            raise ValueError("No eligible positive training weight")
        self.ratios = {key: resolved[key] / total for key in sorted(counts)}
        self.counts = dict(counts)
        weights = [self.ratios[row["slice"]] / counts[row["slice"]] for row in dataset.rows]
        self.weights = torch.tensor(weights, dtype=torch.double)
        if not torch.isfinite(self.weights).all() or (self.weights < 0).any():
            raise ValueError("Mixture weights must be nonnegative and finite")
        requested = config.samples_per_epoch or len(dataset)
        if global_batch_size % world_size:
            raise ValueError("Global batch must be divisible by student DP size")
        # Draw complete optimizer batches so the scheduler never consumes a short tail.
        self.num_samples = (
            math.ceil(requested / global_batch_size) * global_batch_size // world_size
        )
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
