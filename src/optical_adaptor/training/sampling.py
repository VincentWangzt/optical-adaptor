from __future__ import annotations

import math
import random
from dataclasses import dataclass

from optical_adaptor.training.config import fingerprint


@dataclass(frozen=True)
class TaskWindow:
    continuation: list[dict]
    reconstruction: list[dict]
    continuation_tokens: int
    reconstruction_tokens: int
    global_pairs: int


def task_permutations(size: int, seed: int, epoch: int) -> tuple[list[int], list[int]]:
    orders = []
    for task in ("continuation", "reconstruction"):
        order = list(range(size))
        random.Random(int(fingerprint([seed, epoch, task]), 16)).shuffle(order)
        orders.append(order)
    return orders[0], orders[1]


def epoch_windows(
    records: list[dict],
    *,
    seed: int,
    epoch: int,
    global_pairs: int,
    rank: int,
    world_size: int,
    start_update: int = 0,
):
    if not 0 <= rank < world_size or global_pairs % world_size or len(records) % world_size:
        raise ValueError("training pairs and record count must divide evenly across ranks")
    continuation, reconstruction = task_permutations(len(records), seed, epoch)
    for update in range(start_update, math.ceil(len(records) / global_pairs)):
        start = update * global_pairs
        ci, ri = (
            continuation[start : start + global_pairs],
            reconstruction[start : start + global_pairs],
        )
        yield TaskWindow(
            [records[index] for index in ci[rank::world_size]],
            [records[index] for index in ri[rank::world_size]],
            sum(len(records[index]["continuation_ids"]) for index in ci),
            sum(len(records[index]["visual_ids"]) + 1 for index in ri),
            len(ci),
        )
