import re
import unicodedata

LEGAL_SUFFIXES = {
    "limited", "ltd", "llp", "llc", "inc", "incorporated",
    "corp", "corporation", "company", "co", "private", "pvt", "plc"
}


def normalize_text(value):
    if value is None:
        return ""
    value = unicodedata.normalize("NFKD", str(value).lower())
    value = "".join(c for c in value if not unicodedata.combining(c))
    value = value.replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()

def jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def normalize_name(value):
    return normalize_text(value)


def normalize_name_core(value):
    return " ".join(
        t for t in normalize_name(value).split() if t not in LEGAL_SUFFIXES
    )


def normalize_address(value):
    return normalize_text(value)


def tokens(value):
    return set(normalize_text(value).split())


def numeric_tokens(value):
    return set(re.findall(r"\d+", normalize_text(value)))


def char_ngrams(value, n=3):
    text = normalize_text(value).replace(" ", "")
    if not text:
        return set()
    if len(text) <= n:
        return {text}
    return {text[i:i+n] for i in range(len(text) - n + 1)}
