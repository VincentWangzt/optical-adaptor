from __future__ import annotations

import argparse
import sys
from pathlib import Path

from optical_agent.renderer import RenderConfigError, load_render_config, render_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a source-code file into one or more Pillow images."
    )
    parser.add_argument("source", type=Path, help="source-code file to render")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/render.yaml"),
        help="render YAML (default: configs/render.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/rendered"),
        help="directory for page images (default: outputs/rendered)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_render_config(args.config)
        result = render_file(args.source, output_dir=args.output_dir, config=config)
    except (FileNotFoundError, RenderConfigError) as exc:
        print(f"render-code: error: {exc}", file=sys.stderr)
        return 2

    for path in result.page_paths:
        print(path)
    summary = (
        f"Rendered {result.source_line_count} source lines as {result.visual_line_count} "
        f"visual lines on {len(result.page_paths)} page(s)."
    )
    if result.truncated:
        summary += " Output was truncated by pages.max_pages."
    print(summary, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

