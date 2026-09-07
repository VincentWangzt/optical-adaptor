from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from optical_adaptor.training.config import load_pipeline, write_json
from optical_adaptor.training.evaluate import (
    generation_metrics,
    generation_totals,
    refresh_generation_metrics,
)
from optical_adaptor.training.generation import generate_tokens


def test_sampled_generation_replays_without_changing_training_rng():
    pipeline = load_pipeline(Path(__file__).resolve().parents[1] / "configs/training.yaml")
    pipeline = SimpleNamespace(
        config=pipeline.config.model_copy(
            update={
                "evaluation": pipeline.config.evaluation.model_copy(update={"max_new_tokens": 8})
            }
        )
    )
    torch.random.default_generator.manual_seed(1234)
    model = GPT2LMHeadModel(
        GPT2Config(vocab_size=64, n_positions=32, n_embd=16, n_layer=1, n_head=2)
    ).eval()
    qwen = SimpleNamespace(
        device=torch.device("cpu"), assistant_end=63, tokenizer=SimpleNamespace(pad_token_id=62)
    )
    records = [{"record_id": "sample"}]
    ids = torch.tensor([[1, 2]])
    mask = torch.ones_like(ids)
    initial_rng = torch.get_rng_state()
    first = generate_tokens(
        pipeline, qwen, model, records, {"input_ids": ids, "attention_mask": mask}
    )
    assert torch.equal(torch.get_rng_state(), initial_rng)
    torch.rand(128)
    advanced_rng = torch.get_rng_state()
    replay = generate_tokens(
        pipeline, qwen, model, records, {"input_ids": ids, "attention_mask": mask}
    )
    assert torch.equal(replay, first)
    assert torch.equal(torch.get_rng_state(), advanced_rng)
    embedded = generate_tokens(
        pipeline,
        qwen,
        model,
        records,
        {"inputs_embeds": model.get_input_embeddings()(ids), "attention_mask": mask},
    )
    assert torch.equal(first[:, ids.shape[1] :], embedded)
    assert torch.equal(torch.get_rng_state(), advanced_rng)


def test_generation_strata_use_summed_edit_counts(tmp_path):
    records = [
        dict(
            record_id="a",
            split="reconstruction",
            language="Python",
            aspect_ratio=0.7,
            logical_lines=40,
            display_lines=45,
        ),
        dict(
            record_id="b",
            split="reconstruction",
            language="Python",
            aspect_ratio=1.1,
            logical_lines=55,
            display_lines=70,
        ),
        dict(
            record_id="c",
            split="reconstruction",
            language="C",
            aspect_ratio=0.9,
            logical_lines=45,
            display_lines=55,
        ),
    ]
    predictions = [
        dict(
            record_id="a",
            character_distance=2,
            characters=10,
            word_distance=1,
            words=2,
            exact_match=False,
            truncated=False,
        ),
        dict(
            record_id="b",
            character_distance=4,
            characters=90,
            word_distance=3,
            words=18,
            exact_match=False,
            truncated=True,
        ),
        dict(
            record_id="c",
            character_distance=0,
            characters=30,
            word_distance=0,
            words=6,
            exact_match=True,
            truncated=False,
        ),
    ]
    by_id = {row["record_id"]: row for row in records}
    distributed = generation_metrics(
        [
            generation_totals(predictions[::2], by_id),
            generation_totals(predictions[1::2], by_id),
        ]
    )
    assert distributed == generation_metrics([generation_totals(predictions, by_id)])
    assert distributed["records"] == 3
    assert distributed["cer"] == pytest.approx(6 / 130)
    assert distributed["language/Python/cer"] == pytest.approx(0.06)
    assert distributed["language/Python/wer"] == pytest.approx(0.2)
    assert distributed["display_lines/61-80/truncation_rate"] == 1
    assert distributed["aspect/0.75-to-1.0/exact_match"] == 1
    assert distributed["logical_lines/40-49/records"] == 2
    with pytest.raises(ValueError, match="duplicate"):
        generation_totals([predictions[0], predictions[0]], by_id)
    pipeline = SimpleNamespace(config=SimpleNamespace(seed=42))
    write_json(tmp_path / "predictions-000939-rank-0.json", predictions[::2])
    with pytest.raises(ValueError, match="subset"):
        refresh_generation_metrics(pipeline, records, tmp_path, 939, final=True)
    write_json(tmp_path / "predictions-000939-rank-1.json", predictions[1::2])
    assert refresh_generation_metrics(pipeline, records, tmp_path, 939, final=True) == distributed
