from optical_adaptor.edit_distance import evaluate_edit_distance, levenshtein_distance


def test_levenshtein_distance() -> None:
    assert levenshtein_distance("kitten", "sitting") == 3
    assert levenshtein_distance("", "abc") == 3


def test_character_and_word_reports() -> None:
    character = evaluate_edit_distance("abc", "adc", unit="character")
    word = evaluate_edit_distance("one two", "one three", unit="word")

    assert character["distance"] == 1
    assert character["error_rate"] == 1 / 3
    assert word["distance"] == 1
    assert word["reference_units"] == 2
