from reactionfusion.data.preprocessing import detect_language, normalize_text, text_hash


def test_normalize_masks_phone_and_preserves_sinhala() -> None:
    text = "අමතන්න 076 723 3595 දැන්"
    assert normalize_text(text) == "අමතන්න <PHONE> දැන්"


def test_text_hash_normalizes_spacing_and_case() -> None:
    assert text_hash("Test   Post") == text_hash(" test post ")


def test_language_detection() -> None:
    assert detect_language("සිංහල වාක්‍යයක්") == "sinhala"
    assert detect_language("සිංහල post") == "mixed"
    assert detect_language("mata nidi") == "singlish"


def test_masking_normalizes_common_private_identifiers() -> None:
    value = normalize_text("mail me x@example.com or visit https://example.com @name")
    assert value == "mail me <EMAIL> or visit <URL> <USER>"
