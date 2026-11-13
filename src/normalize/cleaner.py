from __future__ import annotations

import html
import re
import unicodedata

# Compile regexes once for performance
WHITESPACE_RE = re.compile(r"\s+")
CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")


def normalize_text(text: str) -> str:
    """Normalize text using Unicode NFKC, unescaping HTML, and removing control chars.

    Preserves numbers, currency symbols ($ € £ ¥), financial terms, and negation.
    """
    if not text:
        return ""

    # 1. Unescape HTML entities (e.g. &amp; -> &, &quot; -> ", &#39; -> ')
    text = html.unescape(text)

    # 2. Unicode NFKC normalization (handles ligatures, full-width characters, non-breaking spaces)
    text = unicodedata.normalize("NFKC", text)

    # 3. Strip dangerous unprintable control characters, but preserve standard whitespace
    text = CONTROL_CHARS_RE.sub("", text)

    # 4. Canonicalize internal whitespace to single standard space
    text = WHITESPACE_RE.sub(" ", text).strip()

    return text


def clean_headline(headline: str) -> str:
    """Cleans a financial news headline while preserving essential financial semantics.

    - Retains negation: 'not', 'no', 'never', 'barely', 'hardly'
    - Retains finance operators/symbols: %, $, bps, +, -
    - Strips clickbait artifacts like leading 'BREAKING:', 'FLASH:', 'UPDATE 1-'
    """
    cleaned = normalize_text(headline)

    # Strip standard wire prefixes commonly attached to headlines
    wire_prefixes = [
        r"^BREAKING\s*:\s*",
        r"^FLASH\s*:\s*",
        r"^EXCLUSIVE\s*:\s*",
        r"^ALERT\s*:\s*",
        r"^UPDATE\s*\d*\s*-\s*",
        r"^CORRECTED\s*-\s*",
        r"^WRAPUP\s*\d*\s*-\s*",
    ]
    for prefix in wire_prefixes:
        cleaned = re.sub(prefix, "", cleaned, flags=re.IGNORECASE).strip()

    return cleaned

