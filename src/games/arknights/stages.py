"""Arknights stage data.

Phase 2: enriched from Penguin Logistics API / PRTS Wiki scraping.
Phase 1: minimal stub with chip schedule and sanity constants.
"""

from __future__ import annotations

# Stage schedule: day of week → open chip stages
CHIP_SCHEDULE: dict[int, list[str]] = {
    0: [],  # Monday
    1: ["PR-C-2"],  # Tuesday — 近卫/特种芯片
    2: [],  # Wednesday
    3: ["PR-C-2"],  # Thursday — 近卫/特种芯片
    4: [],  # Friday
    5: ["PR-C-2"],  # Saturday — 近卫/特种芯片
    6: [],  # Sunday — all open
}

SANITY_CAP = 135  # Max sanity (depends on level)
SANITY_PER_HOUR = 6  # Natural recovery rate
