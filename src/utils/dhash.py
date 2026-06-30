"""Perceptual image hashing via difference hash (dHash).

dHash compares adjacent horizontal pixels — sensitive to the sharp edges
(button borders, text lines, grid dividers) that dominate game UI layouts.
This makes it ideal for "does this screen look like one I've seen before?"
memory matching, as opposed to the MD5-based compute_image_hash which serves
exact deduplication ("is this screenshot byte-identical to the last one?").

Algorithm:
  1. Grayscale conversion
  2. Resize to 9x8 pixels (9 columns × 8 rows → 64 horizontal comparisons → 64-bit)
  3. For each row, compare each pixel to its right neighbor: larger → bit = 1
  4. Pack into a 64-bit integer

Hamming distance ≤ 10 (out of 64) is the standard threshold for "same scene"
with minor variations (animations, scroll position, popup overlays).
"""

from __future__ import annotations

from PIL import Image


def compute_dhash(img: Image.Image) -> int:
    """Compute a 64-bit difference hash for a PIL image.

    Args:
        img: Any PIL image. Converted to grayscale and resized internally.

    Returns:
        64-bit integer hash. Use hamming_distance() to compare two hashes.
    """
    # Convert to grayscale
    if img.mode != "L":
        img = img.convert("L")

    # Resize to 9×8 — captures horizontal gradients with 8px vertical resolution
    small = img.resize((9, 8), Image.LANCZOS)
    pixels = list(small.getdata())

    # Build 64-bit hash: for each row, compare each pixel to its right neighbor
    hash_int = 0
    for y in range(8):
        row_start = y * 9
        for x in range(8):
            if pixels[row_start + x] > pixels[row_start + x + 1]:
                hash_int |= 1 << (y * 8 + x)

    return hash_int


def hamming_distance(h1: int, h2: int) -> int:
    """Hamming distance between two integer hashes (number of differing bits).

    Args:
        h1, h2: 64-bit integers from compute_dhash().

    Returns:
        Integer 0–64. Lower = more similar. ≤10 typically means the same scene.
    """
    return (h1 ^ h2).bit_count()


def dhash_to_hex(h: int) -> str:
    """Convert a 64-bit dHash integer to a 16-character hex string.

    Uses the same hex-string convention as compute_image_hash (hash.py) for
    consistency across the codebase.
    """
    return format(h, "016x")


def hex_to_dhash(hex_str: str) -> int:
    """Convert a 16-character hex string back to a 64-bit dHash integer."""
    return int(hex_str, 16)
