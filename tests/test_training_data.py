from pathlib import Path

import pytest
from pydantic import ValidationError

from optical_adaptor.renderer import load_render_config, render_pages
from optical_adaptor.training.config import PipelineConfig, load_pipeline
from optical_adaptor.training.data import canonicalize, inspection_markup, repository_split

ROOT = Path(__file__).resolve().parents[1]


def test_canonicalization_and_original_offsets():
    source = "a\t= 1  \r\n\r\n\t雪\x00\rZ"
    result = canonicalize(source, frozenset(range(32, 127)), 4)
    assert result.text == "a   = 1\n\n    \\u96ea\\u0000\nZ"
    assert len(result.source_boundaries) == len(result.text) + 1
    index = result.text.index("Z")
    assert source[result.source_boundaries[index] :] == "Z"


def test_inspection_does_not_parse_source_tags():
    assert inspection_markup("<vision>x&y", 8, 11) == ("&lt;vision&gt;<vision>x&amp;y</vision>")
    with pytest.raises(ValueError):
        inspection_markup("x", 2, 3)


def test_repository_assignment_and_schedule_config():
    pipeline = load_pipeline(ROOT / "configs/training.yaml")
    sizes = pipeline.config.data.split_sizes
    assert repository_split("Org/Repo", 42, sizes) == repository_split("org/repo", 42, sizes)
    assert pipeline.config.total_updates == 939
    assert pipeline.config.warmup_updates == 29
    raw = pipeline.config.model_dump()
    raw["training"]["cosine_decay"] = True
    with pytest.raises(ValidationError, match="cosine_decay"):
        PipelineConfig.model_validate(raw)


def test_training_render_fixed_width_auto_height():
    config = load_render_config(ROOT / "configs/render.training.yaml")
    images, lines, truncated = render_pages("x" * 101 + "\n\nend", config=config)
    assert lines == 4
    assert not truncated
    assert len(images) == 1
    assert images[0].width == 1280
    assert images[0].height >= 4 * 28
    more, _, _ = render_pages("\n".join(["x"] * 60), config=config)
    assert more[0].width == images[0].width
    assert more[0].height > images[0].height
