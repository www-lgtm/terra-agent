"""Shared screen → button coordinate cache.

Populated by vlm_describe (VLM sees buttons and their positions)
and consumed by adb_tap (skip full-screen OCR when position is known).

Uses LRU eviction (max 200 entries) to prevent unbounded memory growth
during long-running sessions.

Key: (device_serial, screen_hash, target_lower) → (cx, cy, x1, y1, x2, y2)
     device_serial scopes cache entries to a specific device, preventing
     cross-device contamination when two agents operate simultaneously.
"""

from __future__ import annotations

import logging
from collections import OrderedDict

logger = logging.getLogger(__name__)

MAX_CACHE_SIZE = 200

# Key type: (device_serial, screen_hash, target_lower)
_cache: OrderedDict[tuple[str, str, str], tuple[int, int, int, int, int, int]] = OrderedDict()


def get(screen_hash: str, target: str, device_serial: str = "",
        ocr_texts: list[str] | None = None) -> tuple[int, int, int, int, int, int] | None:
    """Look up a cached button position. Returns None on miss.

    Re-inserts on hit to implement LRU ordering (most recently used at end).
    device_serial scopes the lookup to a specific device (default "" for
    backward compatibility with single-device setups).

    P1: When ocr_texts is provided, tries a composite key (screen_hash + top-5
    OCR texts) first.  Falls back to original screen_hash-only key on miss.
    """
    dev = device_serial or ""
    target_lower = target.lower()

    # P1: composite key lookup (dhash + OCR context) — more precise
    if ocr_texts:
        _top = "_".join(ocr_texts[:5])
        composite_key = (dev, f"{screen_hash}_{_top}", target_lower)
        if composite_key in _cache:
            _cache.move_to_end(composite_key)
            return _cache[composite_key]

    # Fallback: original screen_hash-only key
    key = (dev, screen_hash, target_lower)
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]
    return None


def set(screen_hash: str, target: str, cx: int, cy: int,
        x1: int, y1: int, x2: int, y2: int, device_serial: str = "") -> None:
    """Store a button position, evicting the oldest entry if cache is full."""
    key = (device_serial or "", screen_hash, target.lower())
    if key not in _cache:
        logger.debug("Screen cache: '%s' @ (%d,%d) on %s [dev=%s]", target, cx, cy,
                    screen_hash, device_serial or "default")
    _cache[key] = (cx, cy, x1, y1, x2, y2)
    _cache.move_to_end(key)

    # Evict oldest if over limit
    while len(_cache) > MAX_CACHE_SIZE:
        evicted = _cache.popitem(last=False)
        logger.debug("Screen cache evicted: %s (size=%d)", evicted[0], len(_cache))


def bulk_set(screen_hash: str, buttons: dict[str, tuple[int, int]],
             device_serial: str = "") -> None:
    """Store multiple button → coordinate mappings at once.

    Each coordinate tuple is (cx, cy). We pad an 80x40 bbox around it
    for OCR region cropping during cache verification.
    """
    for name, (cx, cy) in buttons.items():
        pad_x, pad_y = 80, 40
        set(screen_hash, name, cx, cy,
            max(0, cx - pad_x), max(0, cy - pad_y),
            cx + pad_x, cy + pad_y,
            device_serial=device_serial)


def clear() -> None:
    """Clear the entire cache. Useful for tests."""
    _cache.clear()


def size() -> int:
    """Return current cache size. Useful for debugging."""
    return len(_cache)
