import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from optical_adaptor.training.cache import shard_fingerprint, shard_path, verify_shard
from optical_adaptor.training.config import file_sha256, load_pipeline, write_json
from optical_adaptor.training.models import FrozenQwen


def test_shard_fingerprint_shape_dtype_and_corruption(tmp_path):
    pipeline = replace(
        load_pipeline(Path(__file__).resolve().parents[1] / "configs/training.yaml"),
        output=tmp_path,
    )
    records = [{"record_hash": "sample"}]
    path = shard_path(pipeline, "encoder", 0)
    path.parent.mkdir()
    assert not verify_shard(pipeline, "encoder", 0, records)
    save_file({"values": torch.zeros(1, 111, 1280, dtype=torch.bfloat16)}, str(path))
    write_json(
        path.with_suffix(".json"),
        {
            "fingerprint": shard_fingerprint(pipeline, "encoder", records),
            "sha256": file_sha256(path),
        },
    )
    assert verify_shard(pipeline, "encoder", 0, records)
    with pytest.raises(ValueError, match="fingerprint"):
        verify_shard(pipeline, "encoder", 0, [{"record_hash": "different"}])
    metadata = json.loads(path.with_suffix(".json").read_text())
    with path.open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        verify_shard(pipeline, "encoder", 0, records)
    assert metadata["sha256"] != file_sha256(path)


def test_first_and_last_predictions_use_unpadded_positions():
    runtime = object.__new__(FrozenQwen)
    runtime.device = torch.device("cpu")

    def backbone(**kwargs):
        assert not kwargs["use_cache"]
        positions = kwargs["position_ids"]
        assert positions.tolist() == [[0, 1, 2, 3, 4, 5, 6, 6, 6], list(range(9))]
        return SimpleNamespace(last_hidden_state=positions[..., None].float())

    runtime.model = SimpleNamespace(train=lambda _: None, model=backbone)
    hidden = runtime.hidden(
        [torch.ones(7, 1), torch.ones(9, 1)], student=False, target_lengths=[3, 4]
    )
    assert hidden[:, 0].tolist() == [4, 5, 6, 5, 6, 7, 8]
