"""Canonical text shared by rendering, data preparation, and evaluation."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class CanonicalText:
    text: str
    source_boundaries: tuple[int, ...]


def canonicalize(source: str, coverage: frozenset[int], tab_width: int) -> CanonicalText:
    pieces, boundaries = [], [0]
    for match in re.finditer(r"([^\r\n]*)(\r\n|\r|\n|$)", source):
        line, newline = match.groups()
        if not line and not newline:
            continue
        column = 0
        for offset, char in enumerate(line.rstrip(" \t")):
            if char == "\t":
                rendered = " " * (tab_width - column % tab_width)
            elif ord(char) not in coverage or unicodedata.category(char).startswith("C"):
                rendered = f"\\u{ord(char):04x}" if ord(char) <= 0xFFFF else f"\\U{ord(char):08x}"
            else:
                rendered = char
            pieces.append(rendered)
            boundaries.extend([match.start() + offset + 1] * len(rendered))
            column += len(rendered)
        if newline:
            pieces.append("\n")
            boundaries.append(match.end())
    return CanonicalText("".join(pieces), tuple(boundaries))
