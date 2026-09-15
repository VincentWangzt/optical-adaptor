"""Check saved loader positions and preparation identity on the server."""

import json

import pytest
import torch
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader

from optical_adaptor.automodel.config import preparation_fingerprint, read_config
from optical_adaptor.automodel.data import MixtureSampler, validate_preparation


class IndexedRows(Dataset):
    def __init__(self, source):
        self.rows = [
            {"source": source, "task": "reconstruction", "slice": f"{source}/reconstruction/one"}
            for _ in range(24)
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return index


@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("rank", [0, 1])
def test_stateful_loader_continues_same_draws_without_skip_or_repeat(workers, rank):
    _, optical = read_config("configs/automodel.yaml")
    data = IndexedRows(optical.prepare.sources[0].name)

    def loader():
        sampler = MixtureSampler(data, optical.prepare.sources, optical.data, 42, rank, 2)
        return StatefulDataLoader(data, batch_size=2, sampler=sampler, num_workers=workers)

    original = loader()
    iterator = iter(original)
    next(iterator)
    next(iterator)
    snapshot = original.state_dict()
    expected = torch.cat(list(iterator))
    restored = loader()
    restored.load_state_dict(snapshot)
    actual = torch.cat(list(restored))
    assert torch.equal(actual, expected)
    restored.sampler.set_epoch(1)
    original.sampler.set_epoch(1)
    assert torch.equal(torch.cat(list(original)), torch.cat(list(restored)))


def test_reused_data_rejects_changed_preparation_but_allows_sampling_weights(tmp_path):
    raw, optical = read_config("configs/automodel.yaml")
    optical = optical.model_copy(
        update={"prepare": optical.prepare.model_copy(update={"output_dir": str(tmp_path)})}
    )
    (tmp_path / "summary.json").write_text(
        json.dumps({"preparation_fingerprint": preparation_fingerprint(optical, raw["seed"])})
    )
    validate_preparation(optical, raw["seed"])
    sources = optical.prepare.sources
    weighted = optical.model_copy(
        update={
            "prepare": optical.prepare.model_copy(
                update={"sources": [sources[0].model_copy(update={"weight": 3.0}), *sources[1:]]}
            )
        }
    )
    validate_preparation(weighted, raw["seed"])
    limited = optical.model_copy(
        update={
            "prepare": optical.prepare.model_copy(
                update={
                    "sources": [sources[0].model_copy(update={"max_records": 64}), *sources[1:]]
                }
            )
        }
    )
    with pytest.raises(ValueError, match="Prepare a fresh DATA_DIR"):
        validate_preparation(limited, raw["seed"])
