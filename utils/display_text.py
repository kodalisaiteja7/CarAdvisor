"""Normalize user-visible text: no em dashes, en dashes, or hyphen-as-punctuation."""

from __future__ import annotations

import re


def sanitize_user_visible_text(text: str) -> str:
    """Strip dash-like characters from prose shown to users.

    Replaces em/en dashes and spaced hyphens with commas or 'to' for ranges.
    Hyphens between letters (compound words) become a space. Digit hyphen digit
    becomes 'to' (e.g. 0-100 -> 0 to 100). Does not alter URLs or cache keys;
    only apply to narrative fields, not technical identifiers.
    """
    if not text or not isinstance(text, str):
        return text
    t = text
    t = t.replace("—", ", ")
    t = t.replace("–", " to ")
    t = re.sub(r"\s+-\s+", ", ", t)
    # Numeric ranges: 30-50, 0-100 (ASCII hyphen)
    t = re.sub(r"(\d)\s*-\s*(\d)", r"\1 to \2", t)
    # Letter hyphen letter compounds (walk-away -> walk away)
    while re.search(r"[A-Za-z]-[A-Za-z]", t):
        t = re.sub(r"([A-Za-z])-([A-Za-z])", r"\1 \2", t)
    t = re.sub(r",\s*,+", ", ", t)
    t = re.sub(r"\s{2,}", " ", t)
    return t.strip()


def sanitize_nested(obj):
    """Recursively sanitize strings in dicts and lists."""
    if isinstance(obj, str):
        return sanitize_user_visible_text(obj)
    if isinstance(obj, list):
        return [sanitize_nested(x) for x in obj]
    if isinstance(obj, dict):
        return {k: sanitize_nested(v) for k, v in obj.items()}
    return obj
