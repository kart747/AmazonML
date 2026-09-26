"""Text normalization. Raw fields are always kept; these functions create derived views.

Design (RESEARCH_DESIGN.md §3.B): fold accents and transliterate non-Latin scripts
(anyascii), lowercase, unify punctuation, repair common injected noise (junk
prefixes, "m/s", brackets, leetspeak zeros, domains, spaced-out initials), and map a
small hand-written lexicon of street/state abbreviations to one canonical form. No
external data is consulted.
"""
import re
from multiprocessing import Pool

from anyascii import anyascii

from . import config

_RE_DOMAIN = re.compile(r"\b(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9-]*)\.(?:com|in|net|org|co|fr|biz|info|us|io)(?:\.[a-z]{2})?\b")
_RE_MS = re.compile(r"^\W*m/s\b\.?")
_RE_APOS = re.compile(r"[’'`´]")
_RE_LEET0 = re.compile(r"(?<=[a-z])0|0(?=[a-z])")
_RE_NONALNUM = re.compile(r"[^a-z0-9]+")
_RE_INITIALS = re.compile(r"\b(?:[a-z] )+[a-z]\b")
_RE_NULL = re.compile(r"<\s*null\s*>|\bnull\b|\bn/a\b|\bnone\b")
_RE_HOUSE_NO = re.compile(r"\b(?:h\s*\.?\s*no|house\s+no|door\s+no|flat\s+no|plot\s+no|no)\b\.?|\bndeg(?=\d|\b)")
_RE_LEAD0 = re.compile(r"\b0+(?=\d)")

# Canonical street-type / direction forms (US, India, France). Both sides of a pair
# are mapped identically, so the choice of canonical form does not matter.
_ADDR_TOKENS = {
    "street": "st", "str": "st", "road": "rd", "avenue": "ave", "av": "ave", "avenida": "ave",
    "boulevard": "blvd", "bd": "blvd", "boul": "blvd", "drive": "dr", "lane": "ln", "court": "ct",
    "place": "pl", "square": "sq", "highway": "hwy", "parkway": "pkwy", "circle": "cir",
    "terrace": "ter", "trail": "trl", "suite": "ste", "apartment": "apt", "floor": "fl",
    "building": "bldg", "north": "n", "south": "s", "east": "e", "west": "w",
    "r": "rue", "chemin": "ch", "allee": "all", "impasse": "imp", "route": "rte",
    "nagar": "ngr", "marg": "mg", "sector": "sec", "opp": "opposite", "nr": "near",
    # France (test only): "N°" -> anyascii "ndeg" is stripped above; common abbreviations
    "saint": "st", "sainte": "ste", "cours": "crs", "quai": "q",
}
_US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma",
    "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm",
    "new york": "ny", "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
    "district of columbia": "dc",
}
_IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp",
    "jharkhand": "jh", "karnataka": "ka", "kerala": "kl", "madhya pradesh": "mp",
    "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl",
    "odisha": "od", "orissa": "od", "punjab": "pb", "rajasthan": "rj", "sikkim": "sk",
    "tamil nadu": "tn", "telangana": "tg", "tripura": "tr", "uttar pradesh": "up",
    "uttarakhand": "uk", "west bengal": "wb", "delhi": "dl", "new delhi": "dl",
    "jammu and kashmir": "jk", "chandigarh": "ch", "puducherry": "py", "pondicherry": "py",
}
_PHRASES = sorted({**_US_STATES, **_IN_STATES}.items(), key=lambda kv: -len(kv[0]))
_RE_PHRASE = re.compile(r"\b(" + "|".join(re.escape(k) for k, _ in _PHRASES) + r")\b")
_PHRASE_MAP = dict(_PHRASES)


def _glue_initials(s):
    return _RE_INITIALS.sub(lambda m: m.group(0).replace(" ", ""), s)


def norm_name(s: str) -> str:
    """Normalized business name (lowercase ASCII tokens separated by one space)."""
    if not s:
        return ""
    if not s.isascii():
        s = anyascii(s)
    s = s.lower().replace("&", " and ")
    s = _RE_DOMAIN.sub(r" \1 ", s)
    s = _RE_MS.sub(" ", s)
    s = _RE_APOS.sub("", s)
    s = _RE_LEET0.sub("o", s)
    s = _RE_NONALNUM.sub(" ", s)
    s = _glue_initials(" ".join(s.split()))
    return s


def norm_addr(s: str) -> str:
    """Normalized address: ASCII tokens, canonical street/state forms, no house-number prefixes."""
    if not s:
        return ""
    if not s.isascii():
        s = anyascii(s)
    s = s.lower().replace("&", " and ")
    s = _RE_NULL.sub(" ", s)
    s = _RE_APOS.sub("", s)
    s = _RE_NONALNUM.sub(" ", s)
    s = _RE_HOUSE_NO.sub(" ", s)
    s = _RE_LEAD0.sub("", s)
    s = _glue_initials(" ".join(s.split()))
    s = _RE_PHRASE.sub(lambda m: _PHRASE_MAP[m.group(1)], s)
    return " ".join(_ADDR_TOKENS.get(t, t) for t in s.split())


def _map_chunk(args):
    fn, items = args
    f = norm_name if fn == "name" else norm_addr
    return [f(x) for x in items]


def parallel_map(fn: str, values: list, n_proc=None) -> list:
    """Apply norm_name/norm_addr to a list of unique strings with a process pool."""
    n_proc = n_proc or config.N_THREADS
    step = max(1, len(values) // (n_proc * 8) + 1)
    chunks = [(fn, values[i:i + step]) for i in range(0, len(values), step)]
    with Pool(n_proc) as p:
        out = p.map(_map_chunk, chunks)
    return [x for part in out for x in part]
