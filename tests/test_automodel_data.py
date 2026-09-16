"""Focused semantic and resume checks; execute only on the server."""

import json

import pytest
import torch
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader

from optical_adaptor.automodel.config import preparation_fingerprint, read_config
from optical_adaptor.automodel.data import (
    MixtureSampler,
    inherited_values,
    select_evaluation,
    validate_preparation,
)
from optical_adaptor.automodel.prepare import assign_holdouts
from optical_adaptor.automodel.recipe import RunContract


class IndexedRows(Dataset):
    def __init__(self):
        self.rows = [{"slice": "A/reconstruction/images-1"}] * 12 + [
            {"slice": "B/continuation/images-2"}
        ] * 12
        self.all_leaves = sorted({r["slice"] for r in self.rows})

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return index


@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("rank", [0, 1])
def test_loader_resume_accounts_for_prefetch(workers, rank):
    _, optical = read_config("configs/automodel.yaml")
    data = IndexedRows()

    def loader():
        return StatefulDataLoader(
            data,
            batch_size=2,
            sampler=MixtureSampler(data, optical.data, 42, rank, 2, 4),
            num_workers=workers,
        )

    original = loader()
    iterator = iter(original)
    next(iterator)
    next(iterator)
    snapshot = original.state_dict()
    expected = torch.cat(list(iterator))
    restored = loader()
    restored.load_state_dict(snapshot)
    assert torch.equal(torch.cat(list(restored)), expected)


def test_nearest_weight_is_copied_to_leaves_then_normalized():
    leaves = ["A/x/1", "A/x/2", "B/y/1", "B/y/2", "B/y/3"]
    values = inherited_values({"": 1, "A": 3}, leaves)
    assert list(values.values()) == [3, 3, 1, 1, 1]
    assert sum(values[k] for k in leaves if k.startswith("A/")) / sum(
        values.values()
    ) == pytest.approx(2 / 3)
    assert inherited_values({"": 1, "A": 3, "A/x/1": 5}, leaves)["A/x/1"] == 5
    with pytest.raises(ValueError, match="Unknown"):
        inherited_values({"": 1, "typo": 3}, leaves)
    with pytest.raises(ValueError, match="Invalid count"):
        inherited_values({"": 1.5}, leaves, integer=True)


def test_holdouts_are_per_leaf_fixed_and_order_independent():
    rows = [
        {"sample_id": str(i), "slice": "A/x/1" if i < 10 else "B/y/1", "group_id": "same/repo"}
        for i in range(20)
    ]
    first = assign_holdouts(rows, 0.2, 42)
    second = assign_holdouts(list(reversed(rows)), 0.2, 42)
    assert first == second and len(first) == 4
    assert {r["split"] for r in rows} == {"train", "eval"}
    eval_rows = [r for r in rows if r["split"] == "eval"]
    indices, report = select_evaluation(eval_rows, {"A/x/1": 5, "B/y/1": 1}, 42)
    assert len(indices) == 3 and report["A/x/1"]["shortfall"] == 3
    assert set(eval_rows[i]["sample_id"] for i in indices) <= set(first)
    singleton = [{"sample_id": "only", "slice": "C/z/1"}]
    assert assign_holdouts(singleton, 0.2, 42) == []


def test_consumed_counters_resume_and_identity_fails_closed():
    contract = RunContract("a")
    contract.consumed.update({"A/x/1": 5})
    restored = RunContract("a")
    restored.load_state_dict(contract.state_dict())
    assert restored.consumed["A/x/1"] == 5
    with pytest.raises(ValueError, match="contract"):
        RunContract("b").load_state_dict(contract.state_dict())


def test_preparation_identity_excludes_runtime_weights(tmp_path):
    raw, optical = read_config("configs/automodel.yaml")
    optical = optical.model_copy(
        update={"prepare": optical.prepare.model_copy(update={"output_dir": str(tmp_path)})}
    )
    (tmp_path / "summary.json").write_text(
        json.dumps({"preparation_fingerprint": preparation_fingerprint(optical, raw["seed"])})
    )
    weighted = optical.model_copy(
        update={"data": optical.data.model_copy(update={"weights": {"": 3}})}
    )
    validate_preparation(weighted, raw["seed"])
    limited = optical.model_copy(
        update={"prepare": optical.prepare.model_copy(update={"eval_fraction": 0.2})}
    )
    with pytest.raises(ValueError, match="Prepare a fresh DATA_DIR"):
        validate_preparation(limited, raw["seed"])
