from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageColor, ImageDraw, ImageFont


class RenderConfigError(ValueError):
    """Raised when a render YAML is invalid."""


@dataclass(frozen=True)
class Percentage:
    ratio: float


MarginValue = int | Percentage


@dataclass(frozen=True)
class Margins:
    top: MarginValue = 96
    right: MarginValue = 96
    bottom: MarginValue = 96
    left: MarginValue = 96


@dataclass(frozen=True)
class PagesConfig:
    resolution: tuple[int, int | None] | None = (1600, 2200)
    dpi: tuple[int, int] = (144, 144)
    margins: Margins = Margins()
    background: str = "#ffffff"
    max_pages: int | None = None
    start_number: int = 1
    draw_page_number: bool = False


@dataclass(frozen=True)
class TextConfig:
    font: str | None = None
    fallback_fonts: tuple[str, ...] = ()
    font_size: int = 30
    color: str = "#111827"
    line_height: int = 42
    line_width: int = 88
    tab_width: int = 4
    wrap_long_lines: bool = True
    line_numbers: bool = False
    encoding: str = "utf-8"


@dataclass(frozen=True)
class OutputConfig:
    format: str = "PNG"
    filename_pattern: str = "{stem}-page-{page:03d}.{ext}"


@dataclass(frozen=True)
class RenderConfig:
    pages: PagesConfig = PagesConfig()
    text: TextConfig = TextConfig()
    output: OutputConfig = OutputConfig()


@dataclass(frozen=True)
class VisualLine:
    text: str
    source_line: int
    continuation: bool


@dataclass(frozen=True)
class RenderResult:
    page_paths: tuple[Path, ...]
    source_line_count: int
    visual_line_count: int
    truncated: bool


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise RenderConfigError(f"{name} must be a YAML mapping")
    return value


def _only_keys(data: dict[str, Any], allowed: set[str], name: str) -> None:
    unknown = set(data) - allowed
    if unknown:
        rendered = ", ".join(sorted(unknown))
        raise RenderConfigError(f"unknown {name} option(s): {rendered}")


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RenderConfigError(f"{name} must be a positive integer")
    return value


def _pair(value: Any, name: str) -> tuple[int, int]:
    if isinstance(value, int) and not isinstance(value, bool):
        item = _positive_int(value, name)
        return item, item
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise RenderConfigError(f"{name} must be an integer or [width, height]")
    return _positive_int(value[0], f"{name}[0]"), _positive_int(value[1], f"{name}[1]")


def _resolution(value: Any, name: str) -> tuple[int, int | None] | None:
    if isinstance(value, str) and value.strip().lower() == "auto":
        return None
    if isinstance(value, (list, tuple)) and len(value) == 2 and value[1] == "auto":
        return _positive_int(value[0], f"{name}[0]"), None
    return _pair(value, name)


def _margin_value(value: Any, name: str) -> MarginValue:
    if isinstance(value, str) and value.strip().endswith("%"):
        rendered = value.strip()[:-1].strip()
        try:
            percent = float(rendered)
        except ValueError as exc:
            raise RenderConfigError(f"{name} must be a positive integer or percentage") from exc
        if not math.isfinite(percent) or percent <= 0 or percent >= 100:
            raise RenderConfigError(f"{name} percentage must be greater than 0 and below 100")
        return Percentage(percent / 100)
    return _positive_int(value, name)


def _margins(value: Any) -> Margins:
    if value is None:
        raw: dict[str, Any] = {}
    elif isinstance(value, dict):
        raw = value
    elif isinstance(value, (int, str)) and not isinstance(value, bool):
        margin = _margin_value(value, "pages.margins")
        return Margins(top=margin, right=margin, bottom=margin, left=margin)
    else:
        raise RenderConfigError(
            "pages.margins must be a positive integer, percentage, or YAML mapping"
        )

    _only_keys(raw, {"top", "right", "bottom", "left"}, "pages.margins")
    return Margins(
        top=_margin_value(raw.get("top", 96), "pages.margins.top"),
        right=_margin_value(raw.get("right", 96), "pages.margins.right"),
        bottom=_margin_value(raw.get("bottom", 96), "pages.margins.bottom"),
        left=_margin_value(raw.get("left", 96), "pages.margins.left"),
    )


def load_render_config(path: str | Path) -> RenderConfig:
    config_path = Path(path).expanduser().resolve()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except OSError as exc:
        raise RenderConfigError(f"cannot read render config {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise RenderConfigError(f"invalid YAML in {config_path}: {exc}") from exc

    root = _mapping(raw, "config")
    _only_keys(root, {"pages", "text", "output"}, "top-level")
    pages_raw = _mapping(root.get("pages"), "pages")
    text_raw = _mapping(root.get("text"), "text")
    output_raw = _mapping(root.get("output"), "output")

    _only_keys(
        pages_raw,
        {
            "resolution",
            "dpi",
            "margins",
            "background",
            "max_pages",
            "start_number",
            "draw_page_number",
        },
        "pages",
    )
    _only_keys(
        text_raw,
        {
            "font",
            "fallback_fonts",
            "font_size",
            "color",
            "line_height",
            "line_width",
            "tab_width",
            "wrap_long_lines",
            "line_numbers",
            "encoding",
        },
        "text",
    )
    _only_keys(output_raw, {"format", "filename_pattern"}, "output")

    margins = _margins(pages_raw.get("margins"))
    max_pages = pages_raw.get("max_pages")
    if max_pages is not None:
        max_pages = _positive_int(max_pages, "pages.max_pages")

    font = text_raw.get("font")
    if font is not None:
        if not isinstance(font, str) or not font.strip():
            raise RenderConfigError("text.font must be null or a non-empty string")
        candidate = Path(font).expanduser()
        if candidate.is_absolute() or "/" in font or "\\" in font:
            if not candidate.is_absolute():
                candidate = config_path.parent / candidate
            font = str(candidate.resolve())

    fallback_fonts = text_raw.get("fallback_fonts", [])
    if not isinstance(fallback_fonts, list) or not all(
        isinstance(item, str) and item for item in fallback_fonts
    ):
        raise RenderConfigError("text.fallback_fonts must be a list of font paths")
    fallback_fonts = tuple(str((config_path.parent / item).resolve()) for item in fallback_fonts)
    pages = PagesConfig(
        resolution=_resolution(pages_raw.get("resolution", [1600, 2200]), "pages.resolution"),
        dpi=_pair(pages_raw.get("dpi", 144), "pages.dpi"),
        margins=margins,
        background=str(pages_raw.get("background", "#ffffff")),
        max_pages=max_pages,
        start_number=_positive_int(pages_raw.get("start_number", 1), "pages.start_number"),
        draw_page_number=bool(pages_raw.get("draw_page_number", False)),
    )
    text = TextConfig(
        font=font,
        fallback_fonts=fallback_fonts,
        font_size=_positive_int(text_raw.get("font_size", 30), "text.font_size"),
        color=str(text_raw.get("color", "#111827")),
        line_height=_positive_int(text_raw.get("line_height", 42), "text.line_height"),
        line_width=_positive_int(text_raw.get("line_width", 88), "text.line_width"),
        tab_width=_positive_int(text_raw.get("tab_width", 4), "text.tab_width"),
        wrap_long_lines=bool(text_raw.get("wrap_long_lines", True)),
        line_numbers=bool(text_raw.get("line_numbers", False)),
        encoding=str(text_raw.get("encoding", "utf-8")),
    )
    output = OutputConfig(
        format=str(output_raw.get("format", "PNG")).upper(),
        filename_pattern=str(output_raw.get("filename_pattern", "{stem}-page-{page:03d}.{ext}")),
    )
    config = RenderConfig(pages=pages, text=text, output=output)
    _validate_render_config(config)
    return config


def _validate_render_config(config: RenderConfig) -> None:
    margins = config.pages.margins
    _validate_margin_axis(margins.left, margins.right, "horizontal")
    _validate_margin_axis(margins.top, margins.bottom, "vertical")
    if config.pages.resolution is not None:
        width, height = config.pages.resolution
        if height is None:
            height, _, _ = _auto_axis(config.text.line_height, margins.top, margins.bottom)
        resolved = _resolve_fixed_margins(margins, width, height)
        if resolved.left + resolved.right >= width:
            raise RenderConfigError("horizontal margins leave no drawable page width")
        if resolved.top + resolved.bottom >= height:
            raise RenderConfigError("vertical margins leave no drawable page height")
        if height - resolved.top - resolved.bottom < config.text.line_height:
            raise RenderConfigError("page has room for fewer than one line")
    try:
        ImageColor.getrgb(config.pages.background)
        ImageColor.getrgb(config.text.color)
    except ValueError as exc:
        raise RenderConfigError(f"invalid Pillow color: {exc}") from exc
    if config.output.format not in {"PNG", "JPEG", "TIFF", "WEBP"}:
        raise RenderConfigError("output.format must be PNG, JPEG, TIFF, or WEBP")
    try:
        config.output.filename_pattern.format(stem="sample", page=1, ext="png")
    except (KeyError, ValueError) as exc:
        raise RenderConfigError(f"invalid output.filename_pattern: {exc}") from exc


def _margin_ratio(value: MarginValue) -> float:
    return value.ratio if isinstance(value, Percentage) else 0.0


def _validate_margin_axis(before: MarginValue, after: MarginValue, name: str) -> None:
    if _margin_ratio(before) + _margin_ratio(after) >= 1:
        raise RenderConfigError(f"{name} percentage margins must total less than 100%")


def _resolve_margin(value: MarginValue, dimension: int) -> int:
    if isinstance(value, Percentage):
        return math.ceil(value.ratio * dimension)
    return value


def _resolve_fixed_margins(margins: Margins, width: int, height: int) -> Margins:
    return Margins(
        top=_resolve_margin(margins.top, height),
        right=_resolve_margin(margins.right, width),
        bottom=_resolve_margin(margins.bottom, height),
        left=_resolve_margin(margins.left, width),
    )


def _auto_axis(content: int, before: MarginValue, after: MarginValue) -> tuple[int, int, int]:
    content = max(1, content)
    fixed = sum(value for value in (before, after) if isinstance(value, int))
    ratio = _margin_ratio(before) + _margin_ratio(after)
    dimension = max(1, math.ceil((content + fixed) / (1 - ratio)))

    while True:
        before_px = _resolve_margin(before, dimension)
        after_px = _resolve_margin(after, dimension)
        required = content + before_px + after_px
        if required <= dimension:
            return dimension, before_px, after_px
        dimension = required


def _auto_geometry(
    margins: Margins, content_width: int, content_height: int
) -> tuple[tuple[int, int], Margins]:
    width, left, right = _auto_axis(content_width, margins.left, margins.right)
    height, top, bottom = _auto_axis(content_height, margins.top, margins.bottom)
    return (width, height), Margins(top=top, right=right, bottom=bottom, left=left)


def _load_font(config: TextConfig) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if config.font:
        try:
            return ImageFont.truetype(config.font, config.font_size)
        except OSError as exc:
            raise RenderConfigError(f"cannot load text.font {config.font!r}: {exc}") from exc
    return ImageFont.load_default(size=config.font_size)


@lru_cache(maxsize=16)
def font_codepoints(path: str) -> frozenset[int]:
    from fontTools.ttLib import TTFont

    with TTFont(path) as font:
        return frozenset(font.getBestCmap())


class FontChain:
    """Deterministic per-codepoint fallback shared by measurement and drawing."""

    def __init__(self, config: TextConfig):
        self.fonts = [_load_font(config)] + [
            ImageFont.truetype(path, config.font_size) for path in config.fallback_fonts
        ]
        self.coverage = [
            font_codepoints(path)
            for path in (config.font, *config.fallback_fonts)
            if path is not None
        ]

    def runs(self, text: str):
        current, buffer = 0, ""
        for char in text:
            index = next((i for i, cmap in enumerate(self.coverage) if ord(char) in cmap), 0)
            if index != current and buffer:
                yield self.fonts[current], buffer
                buffer = ""
            current, buffer = index, buffer + char
        if buffer:
            yield self.fonts[current], buffer

    def getlength(self, text: str) -> float:
        return sum(font.getlength(run) for font, run in self.runs(text))

    def getbbox(self, text: str) -> tuple[float, float, float, float]:
        x, left, top, right, bottom = 0.0, 0.0, 0.0, 0.0, 0.0
        for font, run in self.runs(text):
            a, b, c, d = font.getbbox(run)
            left, top = min(left, x + a), min(top, b)
            right, bottom = max(right, x + c), max(bottom, d)
            x += font.getlength(run)
        return left, top, max(x, right), bottom

    def draw(self, draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, fill):
        x, y = xy
        for font, run in self.runs(text):
            draw.text((x, y), run, font=font, fill=fill)
            x += font.getlength(run)


def _visual_lines(source: str, config: TextConfig) -> list[VisualLine]:
    source_lines = source.splitlines()
    if not source_lines:
        source_lines = [""]

    output: list[VisualLine] = []
    for number, raw_line in enumerate(source_lines, start=1):
        line = raw_line.expandtabs(config.tab_width)
        if not config.wrap_long_lines or len(line) <= config.line_width:
            output.append(VisualLine(line, number, False))
            continue
        chunks = [
            line[index : index + config.line_width]
            for index in range(0, len(line), config.line_width)
        ]
        output.extend(
            VisualLine(chunk, number, continuation=index > 0) for index, chunk in enumerate(chunks)
        )
    return output


def render_pages(
    source: str,
    *,
    config: RenderConfig,
) -> tuple[list[Image.Image], int, bool]:
    """Render in memory; file and training workflows share identical pixels."""
    font = FontChain(config.text)
    lines = _visual_lines(source, config.text)

    pages = config.pages
    text_config = config.text
    number_width = len(str(max(line.source_line for line in lines)))
    prefix_width = number_width + 2 if text_config.line_numbers else 0

    rendered_lines: list[str] = []
    for visual_line in lines:
        rendered = visual_line.text
        if text_config.line_numbers:
            marker = "" if visual_line.continuation else str(visual_line.source_line)
            rendered = f"{marker:>{number_width}}  {rendered}"
        rendered_lines.append(rendered)

    if pages.resolution is None:
        content_width = math.ceil(
            max(max(font.getlength(line), font.getbbox(line)[2]) for line in rendered_lines)
        )
        content_height = len(lines) * text_config.line_height
        resolution, margins = _auto_geometry(pages.margins, content_width, content_height)
        width, height = resolution
        lines_per_page = len(lines)
    else:
        width, height = pages.resolution
        if height is None:
            height, _, _ = _auto_axis(
                len(lines) * text_config.line_height, pages.margins.top, pages.margins.bottom
            )
        resolution = width, height
        margins = _resolve_fixed_margins(pages.margins, width, height)
        usable_height = height - margins.top - margins.bottom
        lines_per_page = usable_height // text_config.line_height

        widest_sample = "M" * (text_config.line_width + prefix_width)
        measured_width = font.getlength(widest_sample)
        usable_width = width - margins.left - margins.right
        if measured_width > usable_width:
            raise RenderConfigError(
                f"text.line_width={text_config.line_width} needs about "
                f"{math.ceil(measured_width)}px, but only {usable_width}px are available; "
                "lower line_width/font_size or margins"
            )
        for line in rendered_lines:
            left, top, right, bottom = font.getbbox(line)
            if right > usable_width or left < -margins.left:
                raise RenderConfigError("actual glyph extents exceed drawable width")
            if top < 0 or bottom > text_config.line_height:
                raise RenderConfigError("actual glyph extents exceed line height")

    required_pages = math.ceil(len(lines) / lines_per_page)
    rendered_pages = min(required_pages, pages.max_pages or required_pages)
    truncated = rendered_pages < required_pages

    images: list[Image.Image] = []
    for page_index in range(rendered_pages):
        page_number = pages.start_number + page_index
        image = Image.new("RGB", resolution, ImageColor.getrgb(pages.background))
        draw = ImageDraw.Draw(image)
        start = page_index * lines_per_page
        for row, rendered in enumerate(rendered_lines[start : start + lines_per_page]):
            position = (
                margins.left,
                margins.top + row * text_config.line_height,
            )
            font.draw(draw, position, rendered, ImageColor.getrgb(text_config.color))

        if pages.draw_page_number:
            label = str(page_number)
            box = font.getbbox(label)
            label_width = box[2] - box[0]
            draw.text(
                ((width - label_width) / 2, height - margins.bottom / 2),
                label,
                fill=ImageColor.getrgb(text_config.color),
                font=font.fonts[0],
                anchor="mm",
            )

        images.append(image)
    return images, len(lines), truncated


def render_source(
    source: str, *, stem: str, output_dir: str | Path, config: RenderConfig
) -> RenderResult:
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    images, visual_line_count, truncated = render_pages(source, config=config)
    ext = {"JPEG": "jpg"}.get(config.output.format, config.output.format.lower())
    page_paths = []
    for index, image in enumerate(images, start=config.pages.start_number):
        filename = config.output.filename_pattern.format(stem=stem, page=index, ext=ext)
        destination = output_path / filename
        save_options: dict[str, Any] = {"dpi": config.pages.dpi}
        if config.output.format == "PNG":
            save_options["compress_level"] = 6
        image.save(destination, format=config.output.format, **save_options)
        page_paths.append(destination)

    return RenderResult(
        page_paths=tuple(page_paths),
        source_line_count=len(source.splitlines()),
        visual_line_count=visual_line_count,
        truncated=truncated,
    )


def render_file(
    source_path: str | Path,
    *,
    output_dir: str | Path,
    config: RenderConfig,
) -> RenderResult:
    path = Path(source_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"source file does not exist: {path}")
    try:
        source = path.read_text(encoding=config.text.encoding)
    except UnicodeError as exc:
        raise RenderConfigError(
            f"cannot decode {path} with text.encoding={config.text.encoding!r}: {exc}"
        ) from exc
    return render_source(source, stem=path.stem, output_dir=output_dir, config=config)
