import re
import unicodedata

try:
    from unidecode import unidecode
except ImportError:
    unidecode = None


# ============================================================
# LEGAL / BUSINESS SUFFIXES
# ============================================================

LEGAL_SUFFIXES = {
    "limited",
    "ltd",
    "llp",
    "llc",
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "company",
    "co",
    "private",
    "pvt",
    "plc",
    "lp",
}


# Common transliterated / abbreviated Indian business terms.
# These are deliberately conservative.
ABBREVIATIONS = {
    "pvt": "private",
    "pvtltd": "private limited",
    "pvt ltd": "private limited",
    "pra": "private",
    "pra li": "private limited",
    "pra li": "private limited",
    "privaet": "private",
    "ltd": "limited",
}


# ============================================================
# BASIC UNICODE NORMALIZATION
# ============================================================

def unicode_normalize(value):
    """
    Normalize Unicode without deleting non-Latin scripts.
    """

    if value is None:
        return ""

    value = str(value).strip()

    if not value:
        return ""

    value = unicodedata.normalize(
        "NFKC",
        value,
    )

    return value


# ============================================================
# TRANSLITERATION
# ============================================================

def transliterate(value):
    """
    Convert non-Latin scripts into an ASCII representation.

    Example:

        सेवन सन
        ->
        seven sun

    The exact transliteration depends on Unidecode.
    """

    value = unicode_normalize(value)

    if not value:
        return ""

    if unidecode is None:
        return value

    value = unidecode(value)

    return value


# ============================================================
# ENGLISH-LIKE NORMALIZATION
# ============================================================

def normalize_text(value):
    """
    General normalization.

    Important:
    We transliterate BEFORE removing punctuation.
    """

    value = transliterate(value)

    if not value:
        return ""

    value = value.lower()

    # Common punctuation / separators.
    value = value.replace("'", "")
    value = value.replace("’", "")
    value = value.replace("`", "")

    value = value.replace("&", " and ")

    # Everything other than ASCII letters/numbers
    # becomes a space.
    value = re.sub(
        r"[^a-z0-9]+",
        " ",
        value,
    )

    # Collapse whitespace.
    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip()


# ============================================================
# COUNTRY
# ============================================================

def normalize_country(value):

    value = normalize_text(value)

    aliases = {
        "us": "us",
        "usa": "us",
        "united states": "us",
        "united states of america": "us",

        "india": "india",
        "in": "india",
    }

    return aliases.get(
        value,
        value,
    )


# ============================================================
# NAME
# ============================================================

def normalize_name(value):
    return normalize_text(value)


# ============================================================
# NAME CORE
# ============================================================

def normalize_name_core(value):

    name = normalize_name(value)

    if not name:
        return ""

    parts = name.split()

    parts = [
        token
        for token in parts
        if token not in LEGAL_SUFFIXES
    ]

    return " ".join(parts)


# ============================================================
# ADDRESS
# ============================================================

def normalize_address(value):
    return normalize_text(value)


# ============================================================
# TOKENIZATION
# ============================================================

def tokens(value):

    value = normalize_text(value)

    if not value:
        return set()

    return set(
        value.split()
    )


# ============================================================
# ADDRESS TOKENIZATION
# ============================================================

COMMON_ADDRESS_TOKENS = {
    "road",
    "rd",
    "street",
    "st",
    "avenue",
    "ave",
    "boulevard",
    "blvd",
    "drive",
    "dr",
    "lane",
    "ln",
    "way",
    "highway",
    "hwy",
    "place",
    "pl",
    "park",
    "building",
    "bldg",
    "floor",
    "fl",
    "unit",
    "suite",
    "ste",
    "apt",
    "apartment",
    "block",
    "district",
    "city",
    "town",
    "township",
    "county",
    "state",
    "india",
    "usa",
    "us",
    "united",
    "states",
    "north",
    "south",
    "east",
    "west",
    "new",
}


def address_tokens(value):

    address = normalize_address(value)

    if not address:
        return set()

    result = set()

    for token in address.split():

        if len(token) < 3:
            continue

        if token in COMMON_ADDRESS_TOKENS:
            continue

        result.add(token)

    return result


def address_location_tokens(value):

    address = normalize_address(value)

    if not address:
        return set()

    result = set()

    for token in address.split():

        if len(token) < 4:
            continue

        if token in COMMON_ADDRESS_TOKENS:
            continue

        if token.isdigit():
            continue

        result.add(token)

    return result


# ============================================================
# NUMERIC TOKENS
# ============================================================

def numeric_tokens(value):

    value = normalize_text(value)

    if not value:
        return set()

    return set(
        re.findall(
            r"\d+",
            value,
        )
    )


# ============================================================
# CHARACTER N-GRAMS
# ============================================================

def char_ngrams(
    value,
    n=3,
):

    value = normalize_text(value)

    value = value.replace(
        " ",
        "",
    )

    if not value:
        return set()

    if len(value) <= n:
        return {value}

    return {
        value[i:i + n]
        for i in range(
            len(value) - n + 1
        )
    }


# ============================================================
# TOKEN PAIRS
# ============================================================

def token_pairs(value):

    value = normalize_name(value)

    if not value:
        return set()

    token_list = sorted(
        set(value.split())
    )

    result = set()

    for i in range(
        len(token_list)
    ):

        for j in range(
            i + 1,
            len(token_list),
        ):

            result.add(
                (
                    token_list[i],
                    token_list[j],
                )
            )

    return result


# ============================================================
# JACCARD
# ============================================================

def jaccard(a, b):

    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    return len(
        a & b
    ) / len(
        a | b
    )


def token_jaccard(
    text1,
    text2,
):

    return jaccard(
        tokens(text1),
        tokens(text2),
    )


def char_ngram_jaccard(
    text1,
    text2,
    n=3,
):

    return jaccard(
        char_ngrams(text1, n),
        char_ngrams(text2, n),
    )


# ============================================================
# DEBUG
# ============================================================

def show_normalization(value):

    print(
        "Original      :",
        value,
    )

    print(
        "Transliterated:",
        transliterate(value),
    )

    print(
        "Normalized    :",
        normalize_name(value),
    )

    print(
        "Core          :",
        normalize_name_core(value),
    )

    print()


# ============================================================
# TESTS
# ============================================================

def _run_tests():

    print(
        "=" * 70
    )

    print(
        "NORMALIZATION TEST"
    )

    print(
        "=" * 70
    )

    examples = [
        "Seven Sun Producer Private Limited",
        "सेवन सन प्रोड्यूसर प्राइवेट लिमिटेड",
        "Shivam Developers Pvt Ltd",
        "शिवम डेवलपर्स प्रा. लि.",
        "Digital Infra Limited",
        "ಡಿಜಿಟಲ್ ಇನ್‌ಫ್ರಾ ಲಿಮಿಟೆಡ್",
        "Bombay Infra Private Limited",
        "ಬಾಂಬೆ ಇನ್‌ಫ್ರಾ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್",
        "Star Swastik Infotech Private Limited",
        "स्टार स्वस्तिक इंफोटेक प्राइवेट लिमिटेड",
    ]

    for value in examples:

        show_normalization(
            value
        )

    assert normalize_country(
        "USA"
    ) == "us"

    assert normalize_country(
        "India"
    ) == "india"

    assert normalize_name(
        None
    ) == ""

    assert (
        normalize_name_core(
            "ABC Technologies Pvt Ltd"
        )
        == "abc technologies"
    )

    assert (
        jaccard(
            {"a", "b"},
            {"b", "c"},
        )
        == 1 / 3
    )

    print(
        "All normalization tests passed."
    )


if __name__ == "__main__":
    _run_tests()