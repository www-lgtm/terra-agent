"""ScrollTracker — swipe direction + screen-change awareness for loop agent.

The LLM has no persistent memory of "which direction am I scrolling" across
iterations — it re-derives direction from scratch each turn, and gets it wrong
roughly half the time on long tasks.  ScrollTracker bridges this gap by tracking
state in code and injecting plain, actionable hints into the conversation.

Responsibilities:
  - Track current swipe direction and consecutive same-direction count
  - Compare pre-swipe / post-swipe OCR sets to detect list boundaries
  - Edge-fingerprint tracking for grid/table layouts (operator lists, etc.)
  - Inject boundary warnings (1× no-change = warn, 2× = force-flip)
  - Inject periodic progress summaries so the LLM doesn't lose track
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# OCR markers that indicate a horizontal grid/list where scrolling makes sense.
# Used to lower the overlap threshold (grids have lots of repeated UI chrome).
_GRID_MARKERS = [
    # Arknights operator list
    {"等级", "稀有度", "职业", "晋升"},
    {"等级", "稀有度", "自定义排序", "职业"},
    # Generic grid patterns
    {"排序", "筛选"},
]

# Minimum number of markers that must appear in OCR to detect grid context.
_GRID_MARKER_MIN_HITS = 2


@dataclass
class ScrollTracker:
    """Per-task scroll state.  One instance per task execution; reset on new task."""

    direction: str | None = None          # "left" | "right" | "up" | "down" | None
    same_dir_count: int = 0               # consecutive swipes in current direction
    last_ocr_set: frozenset | None = None # OCR texts captured BEFORE the swipe
    unchanged_count: int = 0              # consecutive post-swipe OCRs with ≈0 change
    last_pre_hash: str | None = None      # screen hash captured BEFORE the swipe

    # ── Edge fingerprint (for grid/table list scanning) ───────────
    # Tracks the N rightmost (or leftmost) OCR texts across scrolls.
    # If the edge fingerprint doesn't change for 2+ consecutive scrolls,
    # we've hit the list boundary — even when OCR overlap is high.
    _edge_fingerprints: list[frozenset] = field(default_factory=list)
    _edge_unchanged_count: int = 0
    _grid_context: bool = False  # True when OCR looks like a grid/table listing

    # ── Public API ──────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset all state for a new task."""
        self.direction = None
        self.same_dir_count = 0
        self.last_ocr_set = None
        self.unchanged_count = 0
        self.last_pre_hash = None
        self._edge_fingerprints.clear()
        self._edge_unchanged_count = 0
        self._grid_context = False

    def record_pre_swipe(self, direction: str, ocr_texts: list[str],
                         pre_hash: str | None) -> None:
        """Call BEFORE each adb_swipe or adb_scroll executes.

        Tracks direction continuity — if direction changed, resets
        unchanged_count (the new direction hasn't been tested yet).
        """
        if direction == self.direction:
            self.same_dir_count += 1
        else:
            if self.direction is not None:
                logger.info(
                    "ScrollTracker: dir %s→%s (was %d×, unchanged=%d)",
                    self.direction, direction,
                    self.same_dir_count, self.unchanged_count,
                )
            self.direction = direction
            self.same_dir_count = 1
            self.unchanged_count = 0
            self._edge_fingerprints.clear()
            self._edge_unchanged_count = 0
        self.last_ocr_set = frozenset(ocr_texts) if ocr_texts else None
        self.last_pre_hash = pre_hash

    def analyze_post_swipe(self, post_hash: str | None,
                           ocr_texts: list[str]) -> str | None:
        """Call AFTER screen injection following a swipe.

        Compares pre/post OCR to detect list boundaries.  Returns a
        Chinese system hint string, or None if nothing noteworthy.

        Detection thresholds:
          - Hash unchanged (identical screenshot) → immediate "no effect" warning
          - Edge fingerprint unchanged 2× → boundary confirmed (grid mode)
          - OCR set unchanged → increment unchanged_count
          - OCR overlap >85% (70% for grids) → treated as "no meaningful change"
          - unchanged_count ≥ 2 → boundary confirmed
        """
        current_ocr = frozenset(ocr_texts) if ocr_texts else None

        # ── Detect grid context (operator list, shop grid, etc.) ──
        if not self._grid_context and ocr_texts:
            self._grid_context = self._is_grid_context(ocr_texts)
            if self._grid_context:
                logger.info(
                    "ScrollTracker: grid context detected (overlap threshold lowered to 70%%)"
                )

        # Hash-level check: identical screenshot = swipe had zero effect.
        hash_unchanged = bool(
            post_hash and self.last_pre_hash and post_hash == self.last_pre_hash
        )
        if hash_unchanged:
            self.unchanged_count += 1
        else:
            # Compare OCR sets to decide if the list actually scrolled
            if self.last_ocr_set is not None and current_ocr is not None:
                if current_ocr == self.last_ocr_set:
                    self.unchanged_count += 1
                elif len(current_ocr) > 0:
                    overlap = len(self.last_ocr_set & current_ocr)
                    denom = max(len(self.last_ocr_set), 1)
                    # Grids have lots of repeated chrome → use lower threshold
                    threshold = 0.70 if self._grid_context else 0.85
                    if overlap / denom > threshold:
                        self.unchanged_count += 1
                    else:
                        self.unchanged_count = 0

        # ── Edge fingerprint check (grid context only) ──────────
        edge_boundary = False
        if self._grid_context and ocr_texts and self.direction:
            edge_boundary = self._check_edge_fingerprint(
                ocr_texts, self.direction
            )

        # ── Build hints ─────────────────────────────────────────
        # Priority: hash-identical > confirmed boundary > grid-progress > soft-warning

        if hash_unchanged and self.unchanged_count == 1:
            return (
                f"[系统 — 滑动反馈] 向 {self.direction} 滑动后屏幕完全没变 "
                f"(hash={post_hash[:8]} 未变)。"
                f"已经到达列表边缘。换方向（{'right' if self.direction == 'left' else 'left'}）。"
            )

        if self.unchanged_count >= 2 or edge_boundary:
            opposite = {"left": "right", "right": "left", "up": "down", "down": "up"}
            rev = opposite.get(self.direction, "")
            detail = "OCR 连续不变" if self.unchanged_count >= 2 else "边缘内容连续未更新"
            return (
                f"[系统 — 滑动边界] ★ 向 {self.direction} 方向滑动 "
                f"已到达列表边界（{detail}）。立即改为向 {rev} 方向滑动，"
                f"不要再向 {self.direction} 滑。"
            )

        if self.unchanged_count == 1:
            # Grid context: high overlap is NORMAL for small swipes in grids.
            # Don't say "barely changed" — that misleads the LLM into thinking
            # it hit a boundary when it's actually making progress.
            if self._grid_context:
                return (
                    f"[系统 — 滑动提示] 本次向 {self.direction} 滑动有重叠但这是**"
                    f"小幅度滚动的正常现象，不是边界。** 继续同方向滑，不要反方向。"
                    f"只有连续 2 次完全不变才到达边界。"
                )
            return (
                f"[系统 — 滑动提示] 本次向 {self.direction} 滑动后 "
                f"屏幕内容几乎没有变化。如果下一次依然不变 → 到达边界，换方向。"
            )

        return None

    # ── Internal helpers ───────────────────────────────────────────

    def _is_grid_context(self, ocr_texts: list[str]) -> bool:
        """Detect if current screen is a grid/table listing (operator list, etc.).

        Grids have lots of repeated UI elements (level numbers, sort buttons)
        that inflate OCR overlap — we lower the threshold to avoid false
        negatives at the list boundary.
        """
        ocr_set = set(ocr_texts)
        for markers in _GRID_MARKERS:
            hits = len(markers & ocr_set)
            if hits >= _GRID_MARKER_MIN_HITS:
                return True
        return False

    def _check_edge_fingerprint(
        self, ocr_texts: list[str], direction: str
    ) -> bool:
        """Track the edge OCR texts across scrolls to detect boundary.

        For horizontal grids: extract rightmost (direction='left') or
        leftmost (direction='right') texts and compare across scrolls.
        Returns True if the edge fingerprint hasn't changed for 2+ scrolls.

        Uses the ORDERED ocr_texts list (reading order), NOT a frozenset,
        because edge boundary detection depends on which texts appear at
        the far-left or far-right of the visible grid.
        """
        # Build fingerprint from the trailing N OCR texts.
        # OCR engine returns texts in reading order (L→R, T→B), so the
        # tail of the list is the rightmost/bottommost content on screen.
        edge_n = 8  # How many trailing texts to include in the fingerprint
        if len(ocr_texts) > edge_n:
            tail_texts = ocr_texts[-edge_n:]
        else:
            tail_texts = ocr_texts
        fingerprint = frozenset(tail_texts)

        self._edge_fingerprints.append(fingerprint)
        # Keep only the last 3 fingerprints
        if len(self._edge_fingerprints) > 3:
            self._edge_fingerprints.pop(0)

        if len(self._edge_fingerprints) >= 3:
            # Check if the last 3 fingerprints are all the same
            last3 = self._edge_fingerprints[-3:]
            if last3[0] == last3[1] == last3[2]:
                self._edge_unchanged_count += 1
            else:
                self._edge_unchanged_count = 0

            if self._edge_unchanged_count >= 2:
                return True

        return False

    def build_progress_hint(self) -> str | None:
        """Periodic reminder when many swipes have been done in one direction.

        Returns a hint every ~4 swipes after the 6th swipe, or None.
        In grid context, includes edge fingerprint status.
        """
        if self.same_dir_count >= 6 and self.same_dir_count % 4 == 2:
            base = (
                f"[系统 — 滑动进度] 当前方向: {self.direction}，"
                f"已连续向同方向滑动 {self.same_dir_count} 次。"
                f"请检查最近两次截图的 OCR 文本是否完全相同 → "
                f"相同 = 已到头需换方向，不同 = 继续同方向。"
            )
            if self._grid_context:
                base += (
                    " 当前为列表/网格扫描模式——边缘干员名连续2次未变化也意味着到底。"
                    " 宁可多滑一屏也不要因为\"应该到底了\"就提前换方向。"
                )
            return base
        return None
