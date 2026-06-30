"""Shared FTS5 helpers for CJK-aware search term building.

SQLite FTS5's unicode61 tokenizer splits CJK characters one-by-one, so
AND-ing them fails on any extra character not in the indexed document.
We build CJK bigrams and alphanumeric tokens, then OR them together.
"""

from __future__ import annotations

import re


def build_search_terms(query: str) -> list[str]:
    """Extract meaningful search terms from a user query.

    CJK characters are tokenized individually by unicode61, so AND-ing them
    fails on any extra char not in the indexed document. Instead we build CJK
    bigrams and keep alphanumeric tokens, then OR them in the FTS5 query.

    Args:
        query: Raw user query string.

    Returns:
        List of search terms (lowercased alphanumeric + CJK bigrams).
    """
    terms: list[str] = []

    # Alphanumeric tokens: "1-7", "ep01", "x6"
    for m in re.finditer(r'[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*', query):
        terms.append(m.group().lower())

    # CJK bigrams — covers Unified Ideographs, Extensions A+, Compatibility
    cjk_chars: list[str] = []
    for ch in query:
        cp = ord(ch)
        if (0x4E00 <= cp <= 0x9FFF   # CJK Unified Ideographs
                or 0x3400 <= cp <= 0x4DBF   # Ext-A
                or 0x20000 <= cp <= 0x2A6DF  # Ext-B
                or 0xF900 <= cp <= 0xFAFF  # Compatibility
                or 0x2F800 <= cp <= 0x2FA1F  # Compatibility Supplement
                or 0x2E80 <= cp <= 0x2EFF   # CJK Radicals Supplement
                or 0x3000 <= cp <= 0x303F   # CJK Symbols and Punctuation
                or 0x31C0 <= cp <= 0x31EF   # CJK Strokes
        ):
            cjk_chars.append(ch)
    cjk_str = ''.join(cjk_chars)
    # Single CJK character: add it as a term directly (FTS5 unicode61 tokenizer
    # handles individual CJK chars). Without this, a query with exactly one CJK
    # character produces zero bigrams and returns no results.
    if len(cjk_str) == 1:
        terms.append(cjk_str)
    for i in range(len(cjk_str) - 1):
        bigram = cjk_str[i:i + 2]
        if bigram not in terms:
            terms.append(bigram)

    return terms


def safe_fts5_term(term: str) -> str:
    """Strip characters that break FTS5 query syntax from a single term.

    Args:
        term: A raw search term.

    Returns:
        The term with FTS5-special characters replaced by spaces, stripped.
    """
    return re.sub(r'[\[\]"~@#$,.:;!?/\\&|^<>()\-]', ' ', term).strip()
