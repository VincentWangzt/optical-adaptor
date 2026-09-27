"""Focused checks for batched optical generation inputs and scoring."""

from types import SimpleNamespace

import torch

from optical_adaptor.automodel.config import read_config
from optical_adaptor.automodel.conversations import slice_key
from optical_adaptor.automodel.model import HFGenerationBackend
from optical_adaptor.automodel.processing import collate_generation_inputs
from optical_adaptor.automodel.recipe import OpticalKDRecipe


def sample(tokens, image_position, image_value):
    return {
        "input_ids": torch.tensor([tokens]),
        "attention_mask": torch.ones((1, len(tokens)), dtype=torch.bool),
        "pixel_values": torch.full((1, 1, 1, 1), image_value),
        "image_positions": torch.tensor([[[0, image_position]]]),
    }


def test_generation_collation_left_pads_and_shifts_image_slots():
    inputs = collate_generation_inputs(
        [sample([1, 2, 3], 1, 10), sample([4, 5, 6, 7, 8], 2, 20)], 0
    )
    assert inputs["input_ids"].tolist() == [[0, 0, 1, 2, 3], [4, 5, 6, 7, 8]]
    assert inputs["attention_mask"].tolist() == [
        [False, False, True, True, True],
        [True, True, True, True, True],
    ]
    assert inputs["image_positions"].tolist() == [[[[0, 3]]], [[[1, 2]]]]
    assert inputs["pixel_values"].flatten().tolist() == [10, 20]


def test_hf_backend_replaces_image_embeddings_across_batch():
    class Language:
        def __init__(self):
            self.embedding = torch.nn.Embedding(10, 2)
            self.calls = []

        def get_input_embeddings(self):
            return self.embedding

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            return torch.tensor([[2, 9], [3, 9]])

    backend = object.__new__(HFGenerationBackend)
    backend.language = Language()
    inputs = collate_generation_inputs(
        [sample([1, 2, 3], 1, 10), sample([4, 5, 6, 7, 8], 2, 20)], 0
    )
    image_embeds = torch.tensor([[[11.0, 12.0]], [[21.0, 22.0]]])
    config = SimpleNamespace(max_new_tokens=None, pad_token_id=None, eos_token_id=None)
    ids = backend.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        image_embeds=image_embeds,
        image_positions=inputs["image_positions"],
        generation_config=config,
        max_new_tokens=4,
        pad_token_id=0,
        eos_token_id=9,
    )
    call = backend.language.calls[0]
    assert ids.tolist() == [[2, 9], [3, 9]]
    torch.testing.assert_close(call["inputs_embeds"][0, 3], image_embeds[0, 0])
    torch.testing.assert_close(call["inputs_embeds"][1, 2], image_embeds[1, 0])
    assert call["generation_config"].max_new_tokens == 4
    assert config.max_new_tokens is None


def test_batched_scoring_detects_eos_before_trailing_padding():
    class Tokenizer:
        pad_token_id = 0

        def convert_tokens_to_ids(self, token):
            assert token == "<|im_end|>"
            return 9

        def decode(self, ids, skip_special_tokens):
            assert skip_special_tokens
            return "".join({1: "a", 2: "b"}.get(token, "") for token in ids.tolist())

    class Model:
        tokenizer = Tokenizer()

        def generate(self, **kwargs):
            assert kwargs["input_ids"].shape[0] == 2
            assert kwargs["max_new_tokens"] == 3
            return torch.tensor([[1, 9, 0], [2, 2, 2]])

    recipe = SimpleNamespace(
        dist_env=SimpleNamespace(device=torch.device("cpu")), generation_config=object()
    )
    records = [
        {
            "sample_id": "first",
            "source": "source",
            "task": "reconstruction",
            "image_bin": "1",
            "messages": [{"role": "assistant", "content": "a"}],
        },
        {
            "sample_id": "second",
            "source": "source",
            "task": "continuation",
            "image_bin": "1",
            "messages": [{"role": "assistant", "content": "bb"}],
        },
    ]
    requests = list(
        zip(
            records,
            [sample([1, 2, 3], 1, 10), sample([4, 5, 6, 7, 8], 2, 20)],
            strict=True,
        )
    )
    generation = torch.zeros((2, 6), dtype=torch.float64)
    rows = []
    OpticalKDRecipe._run_generation_batch(
        recipe,
        Model(),
        requests,
        3,
        generation,
        {slice_key(record): i for i, record in enumerate(records)},
        rows,
    )
    assert generation[:, 4:].tolist() == [[1, 0], [1, 1]]
    assert [row["reached_generation_limit"] for row in rows] == [False, True]
    assert [row["prediction"] for row in rows] == ["a", "bbb"]


def test_canonical_generation_batch_size():
    assert read_config("configs/automodel.yaml")[1].evaluation.generation_batch_size == 8
