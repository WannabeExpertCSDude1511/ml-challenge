"""
Normalization utilities for business entity resolution.

Handles:
- Unicode normalization & transliteration (multi-script: Devanagari, Kannada, Tamil, etc.)
- Leet-speak reversal (0→o, 1→l, etc.)
- Zero-padding stripping in addresses
- Legal suffix normalization
- Address component parsing (US & India)
- State normalization (US 2-letter, Indian standard names)
- Phonetic / consonant-skeleton encoding
"""

import re
import unicodedata

try:
    from unidecode import unidecode
except ImportError:
    unidecode = None


# ============================================================
# LEET-SPEAK REVERSAL
# ============================================================

# Map digits that commonly substitute letters in business names.
# Applied BEFORE transliteration so "Lazcan0" → "Lazcano".
_LEET_MAP = str.maketrans({
    "0": "o",
    "1": "l",
    "3": "e",
    "4": "a",
    "5": "s",
    "7": "t",
    "8": "b",
})


def _reverse_leet(value: str) -> str:
    """
    Reverse leet-speak substitutions in a string, but only when the
    digit is surrounded by letters (not in purely numeric tokens).

    "Lazcan0" → "Lazcano"
    "Capita1" → "Capital"
    "1111 Main St" → "1111 Main St" (no change — purely numeric)
    """
    if not value:
        return value

    result = []
    for token in value.split():
        # Only apply leet reversal if the token contains at least
        # one letter — pure numbers like "1111" are addresses.
        if re.search(r"[a-zA-Z]", token):
            result.append(token.translate(_LEET_MAP))
        else:
            result.append(token)
    return " ".join(result)


# ============================================================
# LEGAL / BUSINESS SUFFIXES
# ============================================================

LEGAL_SUFFIXES = {
    "limited", "ltd",
    "llp", "llc",
    "inc", "incorporated",
    "corp", "corporation",
    "company", "co",
    "private", "pvt",
    "plc", "lp",
    "ngo", "trust",
    "foundation",
    "enterprises", "enterprise",
    "solutions", "services",
    "associates",
    "group",
}

# Broader set including Hindi/transliterated abbreviations
LEGAL_SUFFIX_VARIANTS = LEGAL_SUFFIXES | {
    "pvtltd", "pra", "li",
    "privaet", "praivet",
}


# ============================================================
# US STATE NORMALIZATION
# ============================================================

_US_STATE_TO_CODE = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
    "vermont": "VT", "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY", "district of columbia": "DC",
}

# Build reverse lookup (code → code) so both "CA" and "california" → "CA"
_US_CODE_TO_CODE = {v.lower(): v for v in _US_STATE_TO_CODE.values()}


def normalize_us_state(value: str) -> str:
    """Normalize a US state name or abbreviation to 2-letter code."""
    v = value.strip().lower()
    if v in _US_CODE_TO_CODE:
        return _US_CODE_TO_CODE[v]
    return _US_STATE_TO_CODE.get(v, v.upper() if len(v) == 2 else v)


# ============================================================
# INDIAN STATE NORMALIZATION
# ============================================================

_INDIAN_STATE_CANONICAL = {
    "andhra pradesh": "andhra pradesh", "ap": "andhra pradesh",
    "arunachal pradesh": "arunachal pradesh",
    "assam": "assam",
    "bihar": "bihar", "br": "bihar",
    "chhattisgarh": "chhattisgarh", "chattisgarh": "chhattisgarh", "cg": "chhattisgarh",
    "goa": "goa", "ga": "goa",
    "gujarat": "gujarat", "gj": "gujarat",
    "haryana": "haryana", "hr": "haryana",
    "himachal pradesh": "himachal pradesh", "hp": "himachal pradesh",
    "jharkhand": "jharkhand", "jh": "jharkhand",
    "karnataka": "karnataka", "ka": "karnataka",
    "kerala": "kerala", "kl": "kerala",
    "madhya pradesh": "madhya pradesh", "mp": "madhya pradesh",
    "maharashtra": "maharashtra", "mh": "maharashtra",
    "manipur": "manipur", "mn": "manipur",
    "meghalaya": "meghalaya", "ml": "meghalaya",
    "mizoram": "mizoram", "mz": "mizoram",
    "nagaland": "nagaland", "nl": "nagaland",
    "odisha": "odisha", "orissa": "odisha", "od": "odisha", "or": "odisha",
    "punjab": "punjab", "pb": "punjab",
    "rajasthan": "rajasthan", "rj": "rajasthan",
    "sikkim": "sikkim", "sk": "sikkim",
    "tamil nadu": "tamil nadu", "tamilnadu": "tamil nadu", "tn": "tamil nadu",
    "telangana": "telangana", "tg": "telangana", "ts": "telangana",
    "tripura": "tripura", "tr": "tripura",
    "uttar pradesh": "uttar pradesh", "up": "uttar pradesh",
    "uttarakhand": "uttarakhand", "uk": "uttarakhand", "uttaranchal": "uttarakhand",
    "west bengal": "west bengal", "wb": "west bengal",
    "delhi": "delhi", "dl": "delhi", "new delhi": "delhi",
    "chandigarh": "chandigarh", "ch": "chandigarh",
    "puducherry": "puducherry", "pondicherry": "puducherry", "py": "puducherry",
    "jammu and kashmir": "jammu and kashmir", "jk": "jammu and kashmir",
    "ladakh": "ladakh", "la": "ladakh",
    "andaman and nicobar islands": "andaman and nicobar islands",
    "dadra and nagar haveli": "dadra and nagar haveli",
    "daman and diu": "daman and diu",
    "lakshadweep": "lakshadweep",
}


def normalize_indian_state(value: str) -> str:
    """Normalize Indian state name or abbreviation to canonical form."""
    v = value.strip().lower()
    return _INDIAN_STATE_CANONICAL.get(v, v)


# ============================================================
# BASIC UNICODE NORMALIZATION
# ============================================================

def unicode_normalize(value):
    """Normalize Unicode without deleting non-Latin scripts."""
    if value is None:
        return ""
    value = str(value).strip()
    if not value:
        return ""
    value = unicodedata.normalize("NFKC", value)
    return value


# ============================================================
# TRANSLITERATION
# ============================================================

def transliterate(value):
    """
    Convert non-Latin scripts into an ASCII representation.
    Uses unidecode for broad script coverage (Devanagari, Kannada,
    Tamil, Telugu, Gujarati, Malayalam, Bengali, etc.)
    """
    value = unicode_normalize(value)
    if not value:
        return ""
    if unidecode is None:
        return value
    value = unidecode(value)
    return value


# ============================================================
# CORE TEXT NORMALIZATION
# ============================================================

def normalize_text(value):
    """
    General normalization pipeline:
    1. Transliterate non-Latin → ASCII
    2. Lowercase
    3. Remove apostrophes/backticks
    4. Expand & → "and"
    5. Non-alphanumeric → space
    6. Collapse whitespace
    """
    value = transliterate(value)
    if not value:
        return ""
    value = value.lower()
    # Common punctuation / separators
    value = value.replace("'", "")
    value = value.replace("\u2019", "")
    value = value.replace("`", "")
    value = value.replace("&", " and ")
    # Everything other than ASCII letters/numbers → space
    value = re.sub(r"[^a-z0-9]+", " ", value)
    # Collapse whitespace
    value = re.sub(r"\s+", " ", value)
    return value.strip()


# ============================================================
# COUNTRY
# ============================================================

def normalize_country(value):
    value = normalize_text(value)
    aliases = {
        "us": "us", "usa": "us",
        "united states": "us", "united states of america": "us",
        "india": "india", "in": "india",
    }
    return aliases.get(value, value)


# ============================================================
# NAME NORMALIZATION
# ============================================================

def normalize_name(value):
    """Normalize business name: transliterate + clean."""
    return normalize_text(value)


def normalize_name_leet(value):
    """
    Normalize business name with leet-speak reversal.
    Applied after transliteration but before final cleanup.
    "Lazcan0 Clear Trading LLC" → "lazcano clear trading llc"
    """
    value = transliterate(value)
    if not value:
        return ""
    value = _reverse_leet(value)
    value = value.lower()
    value = value.replace("'", "")
    value = value.replace("\u2019", "")
    value = value.replace("`", "")
    value = value.replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def normalize_name_core(value):
    """
    Normalize and strip legal suffixes.
    "ABC Technologies Pvt Ltd" → "abc technologies"
    """
    name = normalize_name_leet(value)
    if not name:
        return ""
    parts = name.split()
    parts = [t for t in parts if t not in LEGAL_SUFFIX_VARIANTS]
    return " ".join(parts)


# ============================================================
# ADDRESS NORMALIZATION
# ============================================================

def normalize_address(value):
    """Normalize address: transliterate + clean."""
    return normalize_text(value)


def normalize_address_stripped(value):
    """
    Normalize address with zero-padding removal.
    "0055 Main St" → "55 main st"
    """
    addr = normalize_address(value)
    if not addr:
        return ""
    # Strip leading zeros from number tokens
    addr = re.sub(r"\b0+(\d+)", r"\1", addr)
    return addr


# ============================================================
# ADDRESS COMPONENT PARSING
# ============================================================

def extract_street_numbers(address: str) -> set:
    """
    Extract all numeric tokens that look like street/house/flat numbers.
    Works on normalized (lowercase ASCII) address text.

    "1111 165th street south holland il" → {"1111", "165"}
    "a 55 vasant kunj new delhi" → {"55"}
    "flat no 02 p no 26 27 30" → {"2", "26", "27", "30"}
    """
    addr = normalize_address_stripped(address)
    if not addr:
        return set()
    # Find digit sequences — including those followed by ordinal suffixes (1st, 2nd, 3rd, 4th...)
    numbers = set()
    for m in re.finditer(r"\b(\d+)(?:st|nd|rd|th)?\b", addr):
        n = m.group(1).lstrip("0") or "0"
        numbers.add(n)
    return numbers


def extract_state(address: str, country: str) -> str:
    """
    Try to extract and normalize the state from an address string.
    Returns normalized state or empty string.
    """
    addr = normalize_address(address)
    if not addr:
        return ""

    country_norm = normalize_country(country)

    if country_norm == "us":
        # Try to find a 2-letter state code at the end or as a standalone token
        tokens_list = addr.split()
        # Check last 1-2 tokens for state
        for i in range(min(3, len(tokens_list))):
            candidate = tokens_list[-(i + 1)]
            result = normalize_us_state(candidate)
            if len(result) == 2 and result != candidate.upper():
                return result
            if result != candidate:
                return result
        # Try multi-word states
        for name, code in _US_STATE_TO_CODE.items():
            if name in addr:
                return code
    elif country_norm == "india":
        # Try known Indian state names/codes in the address
        for name, canonical in _INDIAN_STATE_CANONICAL.items():
            if len(name) > 2 and name in addr:
                return canonical
        # Try 2-letter codes as standalone tokens
        tokens_list = addr.split()
        for token in tokens_list:
            if len(token) == 2:
                result = normalize_indian_state(token)
                if result != token:
                    return result

    return ""


# ============================================================
# TOKENIZATION
# ============================================================

def tokens(value) -> set:
    """Tokenize normalized text into a set of tokens."""
    value = normalize_text(value)
    if not value:
        return set()
    return set(value.split())


def tokens_leet(value) -> set:
    """Tokenize with leet-speak reversal."""
    value = normalize_name_leet(value)
    if not value:
        return set()
    return set(value.split())


# ============================================================
# ADDRESS TOKENIZATION
# ============================================================

COMMON_ADDRESS_TOKENS = {
    "road", "rd", "street", "st", "avenue", "ave",
    "boulevard", "blvd", "drive", "dr", "lane", "ln",
    "way", "highway", "hwy", "place", "pl", "park",
    "building", "bldg", "floor", "fl", "unit", "suite", "ste",
    "apt", "apartment", "block", "district", "city", "town",
    "township", "county", "state",
    "india", "usa", "us", "united", "states",
    "north", "south", "east", "west", "new",
    "no", "plot", "flat", "house", "door", "kh",
}


def address_tokens(value) -> set:
    """Non-common, non-trivial address tokens (len ≥ 3)."""
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


def address_location_tokens(value) -> set:
    """Non-numeric, non-common address tokens (len ≥ 4). Useful for locality/city."""
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

def numeric_tokens(value) -> set:
    """Extract all digit sequences from text, with zero-padding stripped."""
    value = normalize_text(value)
    if not value:
        return set()
    nums = set()
    for n in re.findall(r"\d+", value):
        stripped = n.lstrip("0") or "0"
        nums.add(stripped)
    return nums


# ============================================================
# CHARACTER N-GRAMS
# ============================================================

def char_ngrams(value, n=3) -> set:
    """Character n-grams of normalized text (spaces removed)."""
    value = normalize_text(value)
    value = value.replace(" ", "")
    if not value:
        return set()
    if len(value) <= n:
        return {value}
    return {value[i:i + n] for i in range(len(value) - n + 1)}


# ============================================================
# TOKEN PAIRS (sorted pairs of tokens — order invariant)
# ============================================================

def token_pairs(value) -> set:
    """All sorted pairs of name tokens."""
    value = normalize_name(value)
    if not value:
        return set()
    token_list = sorted(set(value.split()))
    result = set()
    for i in range(len(token_list)):
        for j in range(i + 1, len(token_list)):
            result.add((token_list[i], token_list[j]))
    return result


# ============================================================
# PHONETIC ENCODING
# ============================================================

def consonant_skeleton(value: str) -> str:
    """
    Crude phonetic encoding: keep only consonants, collapse runs.
    "lazcano" → "lzcn"
    "lazcanoo" → "lzcn"
    Good for matching through vowel variations and typos.
    """
    value = normalize_name_leet(value)
    if not value:
        return ""
    # Remove legal suffixes first
    parts = value.split()
    parts = [t for t in parts if t not in LEGAL_SUFFIX_VARIANTS]
    value = "".join(parts)  # No spaces — pure skeleton

    # Keep only consonants
    consonants = re.sub(r"[^bcdfghjklmnpqrstvwxyz]", "", value)
    # Collapse consecutive duplicates
    if not consonants:
        return ""
    result = [consonants[0]]
    for c in consonants[1:]:
        if c != result[-1]:
            result.append(c)
    return "".join(result)


def sorted_token_signature(value: str) -> str:
    """
    Sort name tokens alphabetically (after stripping legal suffixes).
    Handles reordering like "Inc Rojas Capital" ↔ "Rojas Capital Inc".
    """
    core = normalize_name_core(value)
    if not core:
        return ""
    return " ".join(sorted(core.split()))


# ============================================================
# JACCARD HELPERS
# ============================================================

def jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def token_jaccard(text1, text2):
    return jaccard(tokens(text1), tokens(text2))


def char_ngram_jaccard(text1, text2, n=3):
    return jaccard(char_ngrams(text1, n), char_ngrams(text2, n))


# ============================================================
# DEBUG
# ============================================================

def show_normalization(value):
    print("Original       :", value)
    print("Transliterated :", transliterate(value))
    print("Normalized     :", normalize_name(value))
    print("Leet-reversed  :", normalize_name_leet(value))
    print("Core           :", normalize_name_core(value))
    print("Consonant skel :", consonant_skeleton(value))
    print("Sorted tokens  :", sorted_token_signature(value))
    print()


# ============================================================
# SELF-TEST
# ============================================================

def _run_tests():
    print("=" * 70)
    print("NORMALIZATION TESTS")
    print("=" * 70)

    # Basic
    assert normalize_name(None) == ""
    assert normalize_name_core("ABC Technologies Pvt Ltd") == "abc technologies"
    assert normalize_country("USA") == "us"
    assert normalize_country("India") == "india"

    # Leet-speak
    assert normalize_name_leet("Lazcan0") == "lazcano"
    assert normalize_name_leet("Capita1 Partners") == "capital partners"
    assert normalize_name_leet("1111 Main St") == "1111 main st"  # pure numbers untouched

    # Zero-padding
    assert normalize_address_stripped("0055 Main St") == "55 main st"
    assert normalize_address_stripped("002662 Highway 70") == "2662 highway 70"

    # Street numbers
    assert extract_street_numbers("0055 Main St") == {"55"}
    assert "1111" in extract_street_numbers("1111 165th Street")
    assert "165" in extract_street_numbers("1111 165th Street")

    # Consonant skeleton
    assert consonant_skeleton("Lazcano") == "lzcn"
    assert consonant_skeleton("Lynx") == "lynx"

    # Sorted token signature
    assert sorted_token_signature("Inc Rojas Capital Partners") == "capital partners rojas"

    # Jaccard
    assert jaccard({"a", "b"}, {"b", "c"}) == 1 / 3

    # State extraction
    assert normalize_us_state("california") == "CA"
    assert normalize_us_state("CA") == "CA"
    assert normalize_indian_state("MH") == "maharashtra"
    assert normalize_indian_state("tamil nadu") == "tamil nadu"

    print("All normalization tests passed.")


if __name__ == "__main__":
    _run_tests()