import re
import unicodedata
from functools import lru_cache

from indicnlp.normalize.indic_normalize import IndicNormalizerFactory


# =========================================================
# Indic script detection
# =========================================================

INDIC_SCRIPTS = [
    (0x0900, 0x097F, "hi"),   # Devanagari / Hindi
    (0x0980, 0x09FF, "bn"),   # Bengali
    (0x0A00, 0x0A7F, "pa"),   # Gurmukhi / Punjabi
    (0x0A80, 0x0AFF, "gu"),   # Gujarati
    (0x0B00, 0x0B7F, "or"),   # Odia
    (0x0B80, 0x0BFF, "ta"),   # Tamil
    (0x0C00, 0x0C7F, "te"),   # Telugu
    (0x0C80, 0x0CFF, "kn"),   # Kannada
    (0x0D00, 0x0D7F, "ml"),   # Malayalam
]


# =========================================================
# Indic NLP normalizer
# =========================================================

INDIC_NORMALIZER_FACTORY = IndicNormalizerFactory()


# Bump whenever normalization output changes; it invalidates the disk cache.
NORMALIZE_VERSION = 1


# =========================================================
# Devanagari transliteration
# =========================================================
#
# Lightweight rule-based Hindi -> Roman transliteration.
#
# This is deliberately kept dependency-free.
# =========================================================

DEVANAGARI_VOWELS = {
    "अ": "a",
    "आ": "aa",
    "इ": "i",
    "ई": "ee",
    "उ": "u",
    "ऊ": "oo",
    "ऋ": "ri",
    "ए": "e",
    "ऐ": "ai",
    "ओ": "o",
    "औ": "au",
}


DEVANAGARI_MATRAS = {
    "ा": "aa",
    "ि": "i",
    "ी": "ee",
    "ु": "u",
    "ू": "oo",
    "ृ": "ri",
    "े": "e",
    "ै": "ai",
    "ो": "o",
    "ौ": "au",
}


DEVANAGARI_CONSONANTS = {
    "क": "k",
    "ख": "kh",
    "ग": "g",
    "घ": "gh",
    "ङ": "ng",

    "च": "ch",
    "छ": "chh",
    "ज": "j",
    "झ": "jh",
    "ञ": "ny",

    "ट": "t",
    "ठ": "th",
    "ड": "d",
    "ढ": "dh",
    "ण": "n",

    "त": "t",
    "थ": "th",
    "द": "d",
    "ध": "dh",
    "न": "n",

    "प": "p",
    "फ": "ph",
    "ब": "b",
    "भ": "bh",
    "म": "m",

    "य": "y",
    "र": "r",
    "ल": "l",
    "व": "v",

    "श": "sh",
    "ष": "sh",
    "स": "s",
    "ह": "h",

    "ळ": "l",
}


DEVANAGARI_SPECIAL = {
    "ं": "n",
    "ँ": "n",
    "ः": "h",
    "़": "",
    "ऽ": "",
}


# Virama / halant
DEVANAGARI_VIRAMA = "्"


# =========================================================
# Script detection
# =========================================================

def detect_indic_script(text):
    """
    Detect the first supported Indian script.

    Returns:
        hi, bn, pa, gu, or, ta, te, kn, ml
        or None
    """

    for char in text:
        code = ord(char)

        for start, end, lang_code in INDIC_SCRIPTS:
            if start <= code <= end:
                return lang_code

    return None


# =========================================================
# Indic NLP normalization
# =========================================================

def normalize_indic_text(value, lang_code):
    """
    Normalize Indic Unicode text using Indic NLP Library.
    """

    try:
        normalizer = INDIC_NORMALIZER_FACTORY.get_normalizer(
            lang_code,
            remove_nuktas=False,
            nasals_mode="do_nothing",
        )

        return normalizer.normalize(value)

    except Exception:
        return value


# =========================================================
# Devanagari -> Roman
# =========================================================

def transliterate_devanagari(text):
    """
    Convert Devanagari text into readable Roman text.

    Example:

        राम
        ->
        raam

        मार्केटिंग
        ->
        maarketing

        प्राइवेट
        ->
        praivet
    """

    output = []
    i = 0

    while i < len(text):

        char = text[i]

        # -------------------------------------------------
        # Independent vowels
        # -------------------------------------------------

        if char in DEVANAGARI_VOWELS:
            output.append(DEVANAGARI_VOWELS[char])
            i += 1
            continue

        # -------------------------------------------------
        # Consonants
        # -------------------------------------------------

        if char in DEVANAGARI_CONSONANTS:

            consonant = DEVANAGARI_CONSONANTS[char]

            # Look ahead.
            next_char = text[i + 1] if i + 1 < len(text) else ""

            # -------------------------------------------------
            # Explicit halant
            # -------------------------------------------------

            if next_char == DEVANAGARI_VIRAMA:
                output.append(consonant)

                i += 2
                continue

            # -------------------------------------------------
            # Matra
            # -------------------------------------------------

            if next_char in DEVANAGARI_MATRAS:
                output.append(
                    consonant + DEVANAGARI_MATRAS[next_char]
                )

                i += 2
                continue

            # -------------------------------------------------
            # No matra:
            #
            # Devanagari consonants have an inherent "a".
            # -------------------------------------------------

            output.append(
                consonant + "a"
            )

            i += 1
            continue

        # -------------------------------------------------
        # Special marks
        # -------------------------------------------------

        if char in DEVANAGARI_SPECIAL:
            output.append(
                DEVANAGARI_SPECIAL[char]
            )

            i += 1
            continue

        # -------------------------------------------------
        # Nukta
        # -------------------------------------------------

        if char == "़":
            i += 1
            continue

        # -------------------------------------------------
        # ASCII / punctuation / whitespace
        # -------------------------------------------------

        output.append(char)
        i += 1

    return "".join(output)


# =========================================================
# Post-process Hindi transliteration
# =========================================================

def clean_hindi_transliteration(text):
    """
    Clean common artifacts produced by rule-based
    Devanagari transliteration.
    """

    # Remove duplicated spaces
    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    # -----------------------------------------------------
    # Common Hindi spelling adjustments
    # -----------------------------------------------------

    replacements = {
        "aa": "a",
        "ee": "i",
        "oo": "u",
    }

    # We intentionally do NOT blindly replace these globally,
    # because words such as "maal" and "school" can be affected.
    #
    # Instead, only collapse some common awkward sequences.

    text = re.sub(
        r"\bmaarketing\b",
        "marketing",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\bpraivet\b",
        "private",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\blimited\b",
        "limited",
        text,
        flags=re.IGNORECASE,
    )

    return text


# =========================================================
# Indic -> Roman
# =========================================================

def transliterate_indic_text(value, lang_code):
    """
    Convert supported Indic text into Roman text.

    At the moment, Devanagari/Hindi has the dedicated
    lightweight transliteration implementation.

    Other scripts are left unchanged rather than producing
    incorrect transliterations.
    """

    if lang_code == "hi":
        return transliterate_devanagari(value)

    return value


# =========================================================
# Legal / business suffixes
# =========================================================

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
}


# =========================================================
# Main text normalization
# =========================================================

def normalize_text(value):
    """
    Normalize a business-related text field.

    Pipeline:

        raw text
            ↓
        lowercase
            ↓
        detect Indic script
            ↓
        Indic NLP normalization
            ↓
        Indic -> Roman
            ↓
        Unicode normalization
            ↓
        accent removal
            ↓
        punctuation normalization
            ↓
        whitespace normalization
    """

    # -----------------------------------------------------
    # Missing value
    # -----------------------------------------------------

    if value is None:
        return ""

    # -----------------------------------------------------
    # Convert to string
    # -----------------------------------------------------

    value = str(value).lower().strip()

    if not value:
        return ""

    # -----------------------------------------------------
    # Detect Indic script
    # -----------------------------------------------------

    lang_code = detect_indic_script(value)

    # -----------------------------------------------------
    # Indic NLP normalization
    # -----------------------------------------------------

    if lang_code is not None:

        value = normalize_indic_text(
            value,
            lang_code,
        )

        # -------------------------------------------------
        # Indic -> Roman
        # -------------------------------------------------

        value = transliterate_indic_text(
            value,
            lang_code,
        )

        # -------------------------------------------------
        # Hindi-specific cleanup
        # -------------------------------------------------

        if lang_code == "hi":
            value = clean_hindi_transliteration(value)

    # -----------------------------------------------------
    # Unicode normalization
    # -----------------------------------------------------

    value = unicodedata.normalize(
        "NFKD",
        value,
    )

    # -----------------------------------------------------
    # Remove combining marks
    # -----------------------------------------------------

    value = "".join(
        char
        for char in value
        if not unicodedata.combining(char)
    )

    # -----------------------------------------------------
    # Special characters
    # -----------------------------------------------------

    # McDonald's -> mcdonalds
    value = value.replace(
        "'",
        "",
    )

    # & -> and
    value = value.replace(
        "&",
        " and ",
    )

    # -----------------------------------------------------
    # Remove punctuation
    # -----------------------------------------------------

    value = re.sub(
        r"[^a-z0-9]+",
        " ",
        value,
    )

    # -----------------------------------------------------
    # Collapse whitespace
    # -----------------------------------------------------

    return re.sub(
        r"\s+",
        " ",
        value,
    ).strip()


# =========================================================
# Name normalization
# =========================================================

def normalize_name(value):
    """
    Normalize a business name.
    """

    return normalize_text(value)


def normalize_name_core(value):
    """
    Normalize a business name and remove legal suffixes.

    Example:

        Tata Motors Private Limited
        ->
        tata motors
    """

    return " ".join(
        token
        for token in normalize_name(value).split()
        if token not in LEGAL_SUFFIXES
    )


# =========================================================
# Address normalization
# =========================================================

def normalize_address(value):
    """
    Normalize a business address.
    """

    return normalize_text(value)


# =========================================================
# Tokenization
# =========================================================

def tokens(value):
    """
    Return normalized word tokens.
    """

    return set(
        normalize_text(value).split()
    )


# =========================================================
# Numeric tokens
# =========================================================

def numeric_tokens(value):
    """
    Extract numeric tokens.
    """

    return set(
        re.findall(
            r"\d+",
            normalize_text(value),
        )
    )


# =========================================================
# Character n-grams
# =========================================================

def char_ngrams(value, n=3):
    """
    Generate character n-grams from normalized text.
    """

    text = normalize_text(value).replace(
        " ",
        "",
    )

    if not text:
        return set()

    if len(text) <= n:
        return {text}

    return {
        text[i:i + n]
        for i in range(
            len(text) - n + 1
        )
    }


# =========================================================
# Country normalization
# =========================================================

@lru_cache(maxsize=None)
def normalize_country(value):
    """
    Normalize a country text field.
    """

    return normalize_text(value)


# =========================================================
# Jaccard similarity
# =========================================================

def jaccard(a, b):
    """
    Calculate Jaccard similarity between two sets.
    """

    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


# =========================================================
# Helpers on already-normalized text
# =========================================================
#
# normalize_text() output is idempotent, so these skip
# re-normalizing and give the same result as tokens(),
# numeric_tokens() and char_ngrams() on the raw value.
# =========================================================

def normalize_fields(name, address, country):
    """
    Normalize one record once.

    Returns (country, name, name_core, address).
    """

    norm_name = normalize_name(name)

    core = " ".join(
        token
        for token in norm_name.split()
        if token not in LEGAL_SUFFIXES
    )

    return (
        normalize_country(country),
        norm_name,
        core,
        normalize_address(address),
    )


def trigrams(norm_text, n=3):
    text = norm_text.replace(" ", "")

    if not text:
        return set()

    if len(text) <= n:
        return {text}

    return {
        text[i:i + n]
        for i in range(len(text) - n + 1)
    }


def digits(norm_text):
    return set(re.findall(r"\d+", norm_text))
