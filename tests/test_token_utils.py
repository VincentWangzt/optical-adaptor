import pytest

from optical_adaptor.token_utils import count_and_truncate, count_tokens, truncate_to_tokens


class CharacterTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [ord(character) for character in text]

    def decode(self, token_ids: list[int], **_: object) -> str:
        return "".join(chr(token_id) for token_id in token_ids)

    def __call__(self, text: str, **_: object) -> dict[str, list[tuple[int, int]]]:
        return {"offset_mapping": [(index, index + 1) for index in range(len(text))]}


def test_count_and_truncate_returns_both_counts() -> None:
    tokenizer = CharacterTokenizer()
    result = count_and_truncate("abcdef", tokenizer, 4)

    assert result.content == "abcd"
    assert result.original_token_count == 6
    assert result.token_count == 4
    assert result.truncated is True


def test_short_text_is_not_redecoded() -> None:
    tokenizer = CharacterTokenizer()
    result = count_and_truncate("abc", tokenizer, 4)

    assert result.content == "abc"
    assert result.truncated is False
    assert count_tokens(result.content, tokenizer) == 3


def test_zero_limit_and_invalid_limit() -> None:
    tokenizer = CharacterTokenizer()
    assert truncate_to_tokens("abc", tokenizer, 0) == ""
    with pytest.raises(ValueError, match="non-negative"):
        truncate_to_tokens("abc", tokenizer, -1)
