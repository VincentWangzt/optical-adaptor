from pathlib import Path

import pytest
from PIL import Image

from optical_agent.renderer import (
    Margins,
    OutputConfig,
    PagesConfig,
    RenderConfig,
    RenderConfigError,
    TextConfig,
    load_render_config,
    render_source,
)


def test_default_yaml_loads() -> None:
    config = load_render_config(Path("configs/render.yaml"))
    assert config.pages.resolution == (1280, 1280)
    assert config.pages.dpi == (96, 96)
    assert config.text.font_size == 20
    assert config.text.line_height == 28
    assert config.text.line_width == 100


def test_auto_resolution_with_percentage_margins(tmp_path: Path) -> None:
    config_path = tmp_path / "render.yaml"
    config_path.write_text(
        """
pages:
  resolution: auto
  margins: "1%"
text:
  font_size: 10
  line_height: 14
  line_width: 100
output:
  filename_pattern: "{stem}.{ext}"
""".strip()
    )
    config = load_render_config(config_path)
    result = render_source("MMMM\nx", stem="code", output_dir=tmp_path, config=config)

    assert config.pages.resolution is None
    assert len(result.page_paths) == 1
    with Image.open(result.page_paths[0]) as image:
        assert image.size == (38, 30)


def test_default_bundled_font_renders(tmp_path: Path) -> None:
    config = load_render_config(Path("configs/render.yaml"))
    result = render_source("    indented", stem="code", output_dir=tmp_path, config=config)

    assert config.text.font is not None
    assert Path(config.text.font).is_file()
    assert len(result.page_paths) == 1


def test_render_wraps_and_paginates(tmp_path: Path) -> None:
    config = RenderConfig(
        pages=PagesConfig(
            resolution=(500, 260),
            dpi=(100, 100),
            margins=Margins(top=20, right=20, bottom=20, left=20),
        ),
        text=TextConfig(font_size=18, line_height=40, line_width=20),
        output=OutputConfig(filename_pattern="{stem}-{page}.{ext}"),
    )
    source = "x" * 41 + "\n" + "\n".join(f"line {number}" for number in range(5))
    result = render_source(source, stem="code", output_dir=tmp_path, config=config)

    assert result.visual_line_count == 8
    assert len(result.page_paths) == 2
    assert result.page_paths[0].name == "code-1.png"
    with Image.open(result.page_paths[0]) as image:
        assert image.size == (500, 260)
        assert image.info["dpi"] == pytest.approx((100, 100), abs=0.1)


def test_max_pages_marks_result_truncated(tmp_path: Path) -> None:
    config = RenderConfig(
        pages=PagesConfig(
            resolution=(400, 200),
            margins=Margins(top=20, right=20, bottom=20, left=20),
            max_pages=1,
        ),
        text=TextConfig(font_size=16, line_height=40, line_width=20),
    )
    result = render_source(
        "\n".join(["hello"] * 10), stem="code", output_dir=tmp_path, config=config
    )
    assert result.truncated is True
    assert len(result.page_paths) == 1


def test_invalid_available_width_is_actionable(tmp_path: Path) -> None:
    config = RenderConfig(
        pages=PagesConfig(
            resolution=(120, 200),
            margins=Margins(top=10, right=10, bottom=10, left=10),
        ),
        text=TextConfig(font_size=24, line_height=32, line_width=80),
    )
    with pytest.raises(RenderConfigError, match="text.line_width"):
        render_source("hello", stem="code", output_dir=tmp_path, config=config)
