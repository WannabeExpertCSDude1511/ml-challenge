import re
import unicodedata
from functools import lru_cache

from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate


# Bump whenever normalization output changes; it invalidates the disk cache.
NORMALIZE_VERSION = 2


# =========================================================
# Indic scripts
# =========================================================
#
# lang code: (Unicode block start, sanscript scheme,
#             drop word-final inherent "a")
#
# All nine blocks share the same internal layout, so the
# consonant / sign offsets below work for every script.
# =========================================================

INDIC_SCRIPTS = {
    "hi": (0x0900, sanscript.DEVANAGARI, True),
    "bn": (0x0980, sanscript.BENGALI, True),
    "pa": (0x0A00, sanscript.GURMUKHI, True),
    "gu": (0x0A80, sanscript.GUJARATI, True),
    "or": (0x0B00, sanscript.ORIYA, False),
    "ta": (0x0B80, sanscript.TAMIL, False),
    "te": (0x0C00, sanscript.TELUGU, False),
    "kn": (0x0C80, sanscript.KANNADA, False),
    "ml": (0x0D00, sanscript.MALAYALAM, False),
}

INDIC_NORMALIZER_FACTORY = IndicNormalizerFactory()


def _block(base, lo, hi):
    return f"{chr(base + lo)}-{chr(base + hi)}"


# Word-final consonant with no vowel sign / virama after it:
# add a virama so it transliterates without the inherent "a"
# (राम -> ram, not rama).
FINAL_CONSONANT = {}

for _lang, (_base, _, _drop) in INDIC_SCRIPTS.items():
    if _drop:
        _consonant = f"[{_block(_base, 0x15, 0x39)}{_block(_base, 0x58, 0x5F)}]{chr(_base + 0x3C)}?"
        _signs = (
            f"{_block(_base, 0x00, 0x03)}{_block(_base, 0x3C, 0x4D)}"
            f"{_block(_base, 0x55, 0x57)}{_block(_base, 0x62, 0x63)}"
        )
        FINAL_CONSONANT[_lang] = (
            re.compile(rf"({_consonant})(?![{_signs}])(?=\W|$)"),
            chr(_base + 0x4D),
        )


# Signs the transliteration library does not map.
SCRIPT_FIXES = {
    # Devanagari candra vowels (ऑ, ॉ, ॅ): treat as o / e
    "hi": str.maketrans({"ऑ": "ओ", "ॉ": "ो", "ऍ": "ए", "ॅ": "े"}),
    # Malayalam chillu letters -> consonant + virama
    "ml": str.maketrans({
        "ൺ": "ണ്",
        "ൻ": "ന്",
        "ർ": "ര്",
        "ൽ": "ല്",
        "ൾ": "ള്",
        "ൿ": "ക്",
    }),
}


# ITRANS output -> plain lowercase Latin.
ITRANS_FIXES = [
    (re.compile(r"M(?=[pbm])"), "m"),   # anusvara before labials
    ("M", "n"),                          # other anusvara
    (".N", "n"),                         # candrabindu
    ("~N", "n"),
    ("~n", "n"),
    ("R^I", "ri"),
    ("R^i", "ri"),
    ("L^I", "li"),
    ("L^i", "li"),
    ("Ch", "chh"),
    ("H", "h"),
    (".a", ""),
    (".D", "d"),
    ("^", ""),
    ("~", ""),
    (".", ""),
]


# Script-specific spelling fixes after lowercasing.
LATIN_FIXES = {
    # The Tamil scheme renders unvoiced stops as voiced aspirates.
    "ta": [("bh", "p"), ("dh", "t"), ("gh", "k"), ("jh", "s")],
    # Malayalam ṟṟ is pronounced "tt".
    "ml": [("rr", "tt")],
}


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

        if 0x0900 <= code <= 0x0D7F:
            for lang_code, (start, _, _) in INDIC_SCRIPTS.items():
                if start <= code < start + 0x80:
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
# Indic -> Roman
# =========================================================

def transliterate_indic_text(value, lang_code):
    """
    Convert Indic text into lowercase Roman text with the
    indic-transliteration library (ITRANS scheme).

    Example:

        राम मार्केटिंग प्राइवेट लिमिटेड
        ->
        ram marketing praivet limited
    """

    _, scheme, drop_final_a = INDIC_SCRIPTS[lang_code]

    if lang_code in SCRIPT_FIXES:
        value = value.translate(SCRIPT_FIXES[lang_code])

    if drop_final_a:
        pattern, virama = FINAL_CONSONANT[lang_code]
        value = pattern.sub(lambda m: m.group(1) + virama, value)

    value = transliterate(value, scheme, sanscript.ITRANS)

    for old, new in ITRANS_FIXES:
        if isinstance(old, str):
            value = value.replace(old, new)
        else:
            value = old.sub(new, value)

    value = value.lower()

    for old, new in LATIN_FIXES.get(lang_code, ()):
        value = value.replace(old, new)

    return value


# =========================================================
# Canonical tokens
# =========================================================
#
# Variants map to one canonical token, so "Pvt" / "Private"
# / transliterated "praivet" all compare equal. Collisions
# such as street / saint -> "st" are harmless for matching.
# =========================================================

NAME_CANONICAL = {
    # private
    "private": "pvt", "pvt": "pvt", "praivet": "pvt", "praibhet": "pvt",
    "piraivet": "pvt", "praivatt": "pvt",
    # limited
    "limited": "ltd", "ltd": "ltd", "limatid": "ltd", "limitet": "ltd",
    "limittad": "ltd",
    # llp
    "llp": "llp", "elaelapi": "llp", "elelpi": "llp", "ailaailapi": "llp",
    # other legal forms
    "incorporated": "inc", "inc": "inc",
    "corporation": "corp", "corp": "corp",
    "company": "co", "co": "co", "cie": "co", "kampani": "co",
    # common words
    "international": "intl", "intl": "intl",
    "centre": "center", "ctr": "center",
    "services": "service", "svcs": "service", "svc": "service",
    "brothers": "bros",
    "associates": "assoc",
    "manufacturing": "mfg",
    "management": "mgmt",
}

# Abbreviations only trusted when the name was written in an Indic script
# (प्रा. लि. -> pra li); "li" is also a common surname in Latin names.
INDIC_NAME_CANONICAL = {"pra": "pvt", "li": "ltd"}

LEGAL_SUFFIXES = {
    "pvt", "ltd", "llp", "llc", "inc", "corp", "co", "plc",
    "lp", "pllc", "pc", "opc",
    "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc",
    "the",
}

ADDRESS_CANONICAL = {
    "street": "st", "str": "st", "st": "st", "saint": "st",
    "road": "rd", "rd": "rd",
    "avenue": "ave", "ave": "ave", "av": "ave",
    "boulevard": "blvd", "blvd": "blvd", "bd": "blvd", "boul": "blvd",
    "drive": "dr", "dr": "dr",
    "lane": "ln", "ln": "ln",
    "court": "ct", "ct": "ct",
    "place": "pl", "pl": "pl",
    "highway": "hwy", "hwy": "hwy",
    "parkway": "pkwy", "pkwy": "pkwy",
    "square": "sq", "sq": "sq",
    "circle": "cir", "cir": "cir",
    "terrace": "ter", "ter": "ter",
    "trail": "trl", "trl": "trl",
    "suite": "ste", "ste": "ste",
    "apartment": "apt", "apt": "apt",
    "floor": "fl", "flr": "fl", "fl": "fl",
    "building": "bldg", "bldg": "bldg",
    "mount": "mt", "mt": "mt",
    "north": "n", "south": "s", "east": "e", "west": "w",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
    "near": "nr", "nr": "nr",
    "opposite": "opp", "opp": "opp",
    "market": "mkt", "mkt": "mkt",
    "sector": "sec", "sec": "sec",
    "centre": "center", "ctr": "center",
    "number": "no",
    # French
    "chemin": "chem", "chem": "chem",
    "impasse": "imp", "imp": "imp",
    "route": "rte", "rte": "rte",
    "faubourg": "fbg", "fbg": "fbg",
    "allee": "all",
}

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar",
    "california": "ca", "colorado": "co", "connecticut": "ct", "delaware": "de",
    "district of columbia": "dc", "florida": "fl", "georgia": "ga", "hawaii": "hi",
    "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me",
    "maryland": "md", "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne",
    "nevada": "nv", "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm",
    "new york": "ny", "north carolina": "nc", "north dakota": "nd", "ohio": "oh",
    "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa", "rhode island": "ri",
    "south carolina": "sc", "south dakota": "sd", "tennessee": "tn", "texas": "tx",
    "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
}

# Longest names first so "west virginia" wins over "virginia".
US_STATE_PATTERN = re.compile(
    r"\b(" + "|".join(sorted(map(re.escape, US_STATES), key=len, reverse=True)) + r")\b"
)

# "www.acme-foods.com" / "acmefoods.co.in" -> "acmefoods" / "acme-foods"
DOMAIN_PATTERN = re.compile(r"^(?:https?://)?(?:www\.)?([a-z0-9-]+)(?:\.[a-z]{2,})+/?$")


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
    # Indic NLP normalization + Indic -> Roman
    # -----------------------------------------------------

    if lang_code is not None:

        value = normalize_indic_text(
            value,
            lang_code,
        )

        value = transliterate_indic_text(
            value,
            lang_code,
        )

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


def _canonical(text, mapping):
    return " ".join(mapping.get(token, token) for token in text.split())


# =========================================================
# Name normalization
# =========================================================

def normalize_name(value):
    """
    Normalize a business name.

    Domain-style names keep only the domain label, and
    legal / common words are mapped to canonical tokens.

    Example:

        www.maurewilliamscolombier.com -> maurewilliamscolombier
        Tata Motors Private Limited    -> tata motors pvt ltd
    """

    if value is None:
        return ""

    raw = str(value).strip().lower()
    domain = DOMAIN_PATTERN.match(raw)
    if domain:
        raw = domain.group(1)

    text = _canonical(normalize_text(raw), NAME_CANONICAL)

    if detect_indic_script(raw):
        text = _canonical(text, INDIC_NAME_CANONICAL)

    return text


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

    Example:

        85 Wayne Avenue, Ticonderoga, New York -> 85 wayne ave ticonderoga ny
    """

    text = US_STATE_PATTERN.sub(
        lambda m: US_STATES[m.group(1)],
        normalize_text(value),
    )

    return _canonical(text, ADDRESS_CANONICAL)


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

    return trigrams(normalize_text(value), n)


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
