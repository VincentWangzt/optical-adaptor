from types import SimpleNamespace

import pytest

from optical_adaptor.training.config import write_json
from optical_adaptor.training.evaluate import (
    generation_metrics,
    generation_totals,
    refresh_generation_metrics,
)


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
