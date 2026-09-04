from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TruncatedText:
    """Text plus tokenizer counts before and after optional truncation."""

    content: str
    original_token_count: int
    token_count: int
    truncated: bool


def _input_ids(tokenized: Any) -> list[int]:
    if isinstance(tokenized, dict):
        tokenized = tokenized["input_ids"]
    elif hasattr(tokenized, "input_ids"):
        tokenized = tokenized.input_ids
    if hasattr(tokenized, "tolist"):
        tokenized = tokenized.tolist()
    if tokenized and isinstance(tokenized[0], list):
        tokenized = tokenized[0]
    return list(tokenized)


def count_tokens(text: str, tokenizer: Any) -> int:
    """Count text tokens without adding model-level BOS/EOS or chat tokens."""

    return len(tokenizer.encode(text, add_special_tokens=False))


def truncate_to_tokens(text: str, tokenizer: Any, max_tokens: int) -> str:
    """Return the prefix represented by at most ``max_tokens`` tokenizer tokens."""

    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 0:
        raise ValueError("max_tokens must be a non-negative integer")
    token_ids = list(tokenizer.encode(text, add_special_tokens=False))
    if len(token_ids) <= max_tokens:
        return text
    if max_tokens == 0:
        return ""

    # Fast tokenizers can map the token boundary back to an exact source-code prefix.
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offsets = encoded["offset_mapping"]
        if offsets and isinstance(offsets[0], list) and len(offsets[0]) != 2:
            offsets = offsets[0]
        end = int(offsets[max_tokens - 1][1])
        if end > 0:
            return text[:end]
    except (KeyError, NotImplementedError, TypeError, ValueError):
        pass

    # Byte-level model tokenizers round-trip source code exactly in normal use.
    return tokenizer.decode(
        token_ids[:max_tokens],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def count_and_truncate(text: str, tokenizer: Any, max_tokens: int | None) -> TruncatedText:
    """Count ``text`` and optionally return a token-bounded prefix with both counts."""

    original_count = count_tokens(text, tokenizer)
    if max_tokens is None or original_count <= max_tokens:
        return TruncatedText(text, original_count, original_count, False)
    content = truncate_to_tokens(text, tokenizer, max_tokens)
    return TruncatedText(
        content=content,
        original_token_count=original_count,
        token_count=count_tokens(content, tokenizer),
        truncated=True,
    )


def load_tokenizer(
    model_id: str,
    *,
    revision: str | None = None,
    trust_remote_code: bool = False,
) -> Any:
    """Load a Hugging Face tokenizer lazily so text-only utilities stay lightweight."""

    try:
        from transformers import AutoTokenizer, LlamaTokenizerFast, PreTrainedTokenizerFast
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError("tokenizer dependencies are missing; run `uv sync`") from exc
    if "deepseek-ocr" in model_id.lower():
        # Avoid importing DeepSeek's legacy remote model configuration just to
        # read its standard Llama tokenizer files under Transformers 5.
        return LlamaTokenizerFast.from_pretrained(model_id, revision=revision)
    try:
        return AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
    except ImportError:
        if not trust_remote_code:
            raise
        # DeepSeek's tokenizer.json is standard even when its remote model code is
        # incompatible with the active Transformers major version.
        return PreTrainedTokenizerFast.from_pretrained(model_id, revision=revision)
