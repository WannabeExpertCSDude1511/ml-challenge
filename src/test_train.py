from src.normalize import normalize_text

def test_normalization():

    test_cases = [
        # English
        ("ABC Pvt. Ltd.", "abc pvt ltd"),

        # Extra spaces + punctuation
        ("  Tata   Motors, Inc. ", "tata motors inc"),

        # Apostrophe
        ("McDonald's Barbershop", "mcdonalds barbershop"),

        # Accented Latin characters
        ("Café Élite", "cafe elite"),

        # Hindi / Devanagari
        ("राम मार्केटिंग प्राइवेट लिमिटेड", None),

        # Missing value
        (None, ""),
    ]

    for original, expected in test_cases:

        result = normalize_text(original)

        print(f"{original!r} → {result!r}")

        # For Hindi, just verify that transliteration happened.
        if expected is None:
            assert result != ""
            assert not any(
                "\u0900" <= c <= "\u097F"
                for c in result
            )

        else:
            assert result == expected


if __name__ == "__main__":
    test_normalization()
    print("\nAll normalization tests passed!")
