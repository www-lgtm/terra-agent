"""OpenCV template matching for core UI buttons (L1 fast path).

NOTE: template matching is NOT suitable for small pure-icon buttons without
distinctive texture, like the Arknights base notification bell (基建铃铛).
The bell icon is ~40x50px with no unique visual features, and the base screen
is densely packed with similar-shaped elements. Trying templates like
基建消息.png / 制造站收取提醒.png fails reliably — stick to hardcoded
percentage coordinates (adb_tap_position) for the bell.
"""

from __future__ import annotations

import collections
import logging
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from config.settings import config

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(config.DATA_DIR) / "templates"


class TemplateMatcher:
    """Match template images against screenshots using OpenCV matchTemplate.

    Thread-safe: multiple agents can load and match templates concurrently
    on different devices without corrupting the shared _templates dict.
    """

    def __init__(
        self,
        max_cache_entries: int = 500,
        cache_ttl: float = 300.0,
    ) -> None:
        self._lock = threading.Lock()
        self._templates: dict[str, np.ndarray] = {}
        self._position_cache: collections.OrderedDict[
            str, tuple[int, int, float]
        ] = collections.OrderedDict()
        self._max_cache_entries = max_cache_entries
        self._cache_ttl = cache_ttl

    def load_template(self, name: str, image_path: str | Path) -> None:
        """Load a template image for matching.

        Uses numpy.fromfile + cv2.imdecode to support non-ASCII paths
        (e.g. Chinese filenames) on Windows.

        Returns silently if the template is already loaded (idempotent).
        """
        with self._lock:
            if name in self._templates:
                return
        path = Path(image_path)
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Template not found or unreadable: {image_path}")
        with self._lock:
            # Double-check after acquiring lock — another thread may have loaded
            # it between our first check and the I/O above.
            if name in self._templates:
                return
            self._templates[name] = img
            logger.debug("Loaded template '%s' from %s", name, image_path)

    def load_templates_from_dir(self, directory: str | Path) -> int:
        """Load all PNG images from a directory as templates. Returns count loaded."""
        directory = Path(directory)
        count = 0
        for png in directory.glob("*.png"):
            name = png.stem
            self.load_template(name, png)
            count += 1
        return count

    def load_all_templates(self, game: str = "arknights") -> int:
        """Recursively load all PNG templates from data/templates/{game}/.

        Uses the relative path (minus extension) as the template name,
        e.g. 'elite_e0' for elite_e0.png, 'task_done' for task_done.png,
        'elite/e2' for elite/e2.png (when using subdirectories).
        Returns the total number of templates loaded.
        """
        directory = TEMPLATES_DIR / game
        if not directory.exists():
            logger.warning("Template directory not found: %s", directory)
            return 0
        count = 0
        for png in directory.rglob("*.png"):
            # Relative path from game dir, forward slashes, no extension
            name = str(png.relative_to(directory).with_suffix("")).replace("\\", "/")
            self.load_template(name, png)
            count += 1
        logger.info("Loaded %d templates for game '%s' from %s", count, game, directory)
        return count

    # ── Template lifecycle management ──

    def ensure_templates_for_game(self, game: str = "arknights") -> int:
        """Idempotently load all templates for a game. Thread-safe.

        Uses double-checked locking so that:
        - The first call loads templates from disk.
        - Subsequent calls return immediately with the loaded count.
        - Multiple threads can call this concurrently without double-loading.
        """
        # Fast path: unlocked read — if already loaded, return immediately.
        games_loaded = getattr(self, "_games_loaded", None)
        if games_loaded is not None and games_loaded.get(game):
            with self._lock:
                return sum(1 for _ in self._templates)
        # Slow path: may need to load. Set the flag BEFORE releasing lock
        # so no other thread also enters load_all_templates.
        with self._lock:
            if not hasattr(self, "_games_loaded"):
                self._games_loaded: dict[str, bool] = {}
            self._games_loaded[game] = True
        # Load OUTSIDE the lock — load_all_templates -> load_template
        # internally acquires self._lock, and Lock is not reentrant.
        return self.load_all_templates(game)

    def get_template(self, name: str) -> np.ndarray | None:
        """Look up a loaded template by name, trying common naming conventions.

        Transparently handles the discrepancy between icons.json keys
        (e.g. 'material_固源岩') and load_all_templates keys
        (e.g. 'materials/material_固源岩').

        Tries: exact match → with 'materials/' prefix → without prefix.
        """
        with self._lock:
            tpl = self._templates.get(name)
            if tpl is not None:
                return tpl
            if not name.startswith("materials/"):
                return self._templates.get(f"materials/{name}")
            return self._templates.get(name.split("/", 1)[1])

    def match(
        self,
        screenshot: Image.Image | str | Path,
        template_name: str,
        threshold: float | None = None,
        grayscale: bool = False,
        color_weighted: bool = False,
        method: int = cv2.TM_CCOEFF_NORMED,
        multi_scale: bool = False,
        scale_range: tuple[float, float, float] = (0.85, 1.0, 1.15),
    ) -> dict | None:
        """Match a template against a screenshot. Returns {center, score, bbox} or None.

        Args:
            screenshot: PIL image, path, or ndarray.
            template_name: Name of the loaded template.
            threshold: Minimum confidence score (0-1). Default from config.
            grayscale: If True, convert both template and screenshot to grayscale
                       before matching. This eliminates background-color variation
                       (e.g. different-rarity card backgrounds) from the correlation.
            color_weighted: If True, match each BGR channel independently and
                            average the correlation scores. Preserves color info
                            that grayscale matching discards — essential for
                            distinguishing materials with similar shapes but
                            different colors (e.g. 固源岩 vs 固源岩组).
                            Takes precedence over grayscale.
            method: OpenCV matchTemplate method (default TM_CCOEFF_NORMED).
                    TM_CCORR_NORMED can work better for templates with large
                    uniform background regions but is less discriminative.
            multi_scale: If True, try matching the template at multiple scales
                         (defined by scale_range) and return the best result.
                         Compensates for resolution differences between the
                         template capture device and the current device.
            scale_range: Three scale factors to try when multi_scale=True.
                         Default (0.85, 1.0, 1.15) covers ~15% size variation.
        """
        threshold = threshold or config.vision.template_match_threshold
        with self._lock:
            template = self._templates.get(template_name)
        if template is None:
            logger.warning("Template '%s' not loaded", template_name)
            return None
        screen = self._to_cv(screenshot)

        # ── Color-weighted matching (3-channel, preserves color info) ──
        if color_weighted:
            if multi_scale:
                return self._match_color_weighted_multi_scale(
                    screen, template, template_name, threshold, method, scale_range,
                )
            return self._match_color_weighted(
                screen, template, template_name, threshold, method,
            )

        if grayscale:
            if len(screen.shape) == 3:
                screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

        # ── Multi-scale matching ──
        if multi_scale:
            return self._match_multi_scale(
                screen, template, template_name, threshold, grayscale, method, scale_range,
            )

        # ── Single-scale (original behaviour) ──
        return self._match_single(
            screen, template, template_name, threshold, grayscale, method,
        )

    def _match_single(
        self,
        screen: np.ndarray,
        template: np.ndarray,
        template_name: str,
        threshold: float,
        grayscale: bool,
        method: int,
    ) -> dict | None:
        """Core single-scale matchTemplate call."""
        tpl = template
        if grayscale and len(tpl.shape) == 3:
            tpl = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)

        result = cv2.matchTemplate(screen, tpl, method)
        if method in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED):
            # SQDIFF: 0 = perfect match, lower is better.
            # Convert threshold so callers use the same 0-1 scale as CCOEFF:
            # threshold=0.85 → SQDIFF must be ≤ 0.15 to pass.
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            best_val, best_loc = min_val, min_loc
            if best_val > (1.0 - threshold):
                return None
        else:
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            best_val, best_loc = max_val, max_loc
            if best_val < threshold:
                return None

        h, w = tpl.shape[:2]
        return {
            "name": template_name,
            "score": float(best_val),
            "center": (best_loc[0] + w // 2, best_loc[1] + h // 2),
            "bbox": (best_loc[0], best_loc[1], best_loc[0] + w, best_loc[1] + h),
            "scale": 1.0,
        }

    def _match_multi_scale(
        self,
        screen: np.ndarray,
        template: np.ndarray,
        template_name: str,
        threshold: float,
        grayscale: bool,
        method: int,
        scale_range: tuple[float, float, float],
    ) -> dict | None:
        """Try matching at multiple template scales, return best result above threshold."""
        tpl = template
        if grayscale and len(tpl.shape) == 3:
            tpl = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)

        best: dict | None = None
        th, tw = tpl.shape[:2]

        for scale in scale_range:
            if scale == 1.0:
                tpl_scaled = tpl
            else:
                sw = max(6, int(tw * scale))
                sh = max(6, int(th * scale))
                # Skip if template would be larger than screen
                if sw > screen.shape[1] or sh > screen.shape[0]:
                    continue
                tpl_scaled = cv2.resize(tpl, (sw, sh), interpolation=cv2.INTER_AREA)

            result = cv2.matchTemplate(screen, tpl_scaled, method)
            if method in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED):
                min_val, _, min_loc, _ = cv2.minMaxLoc(result)
                best_val, best_loc = min_val, min_loc
                # SQDIFF: 0 = perfect, lower is better. Convert threshold.
                if best_val > (1.0 - threshold):
                    continue
                # Convert to score where higher is better for comparison
                score = 1.0 - best_val  # SQDIFF_NORMED: 0=perfect
            else:
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                best_val, best_loc = max_val, max_loc
                if best_val < threshold:
                    continue
                score = best_val

            sh_, sw_ = tpl_scaled.shape[:2]
            result_dict = {
                "name": template_name,
                "score": float(score),
                "center": (best_loc[0] + sw_ // 2, best_loc[1] + sh_ // 2),
                "bbox": (best_loc[0], best_loc[1], best_loc[0] + sw_, best_loc[1] + sh_),
                "scale": scale,
            }
            if best is None or score > best["score"]:
                best = result_dict

        if best is not None:
            logger.debug(
                "Template '%s' multi_scale best: score=%.4f, scale=%.2f",
                template_name, best["score"], best.get("scale", 1.0),
            )
        return best

    # ── Color-weighted matching (3-channel BGR) ────────────────────────

    def _match_color_weighted(
        self,
        screen: np.ndarray,
        template: np.ndarray,
        template_name: str,
        threshold: float,
        method: int,
    ) -> dict | None:
        """Match BGR channels independently and average correlation scores.

        Unlike grayscale matching (which throws away all color information),
        this preserves the per-channel correlation structure. Materials that
        look identical in grayscale but differ in color (e.g. 固源岩组 greenish
        vs 提纯源岩 purplish) produce different per-channel scores and are
        properly distinguished.

        For TM_CCOEFF_NORMED (default): scores are in [-1, 1], higher = better.
        We average the 3 result maps and find the global maximum.
        """
        if len(screen.shape) == 2 or len(template.shape) == 2:
            # Single-channel input — fall back to standard grayscale match
            return self._match_single(screen, template, template_name, threshold, True, method)

        th, tw = template.shape[:2]
        if th > screen.shape[0] or tw > screen.shape[1]:
            return None

        # Match each BGR channel independently, accumulate into float64 sum
        result_sum: np.ndarray | None = None
        for c in range(3):
            ch_result = cv2.matchTemplate(
                screen[:, :, c], template[:, :, c], method,
            )
            if result_sum is None:
                result_sum = ch_result.astype(np.float64)
            else:
                result_sum += ch_result

        result_avg = result_sum / 3.0  # type: ignore[operator]

        if method in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED):
            # SQDIFF: 0 = perfect, lower is better. Convert threshold.
            min_val, _, min_loc, _ = cv2.minMaxLoc(result_avg)
            if min_val > (1.0 - threshold):
                return None
            return {
                "name": template_name,
                "score": float(1.0 - min_val),
                "center": (min_loc[0] + tw // 2, min_loc[1] + th // 2),
                "bbox": (min_loc[0], min_loc[1], min_loc[0] + tw, min_loc[1] + th),
                "scale": 1.0,
            }
        else:
            _, max_val, _, max_loc = cv2.minMaxLoc(result_avg)
            if max_val < threshold:
                return None
            return {
                "name": template_name,
                "score": float(max_val),
                "center": (max_loc[0] + tw // 2, max_loc[1] + th // 2),
                "bbox": (max_loc[0], max_loc[1], max_loc[0] + tw, max_loc[1] + th),
                "scale": 1.0,
            }

    def _match_color_weighted_multi_scale(
        self,
        screen: np.ndarray,
        template: np.ndarray,
        template_name: str,
        threshold: float,
        method: int,
        scale_range: tuple[float, float, float],
    ) -> dict | None:
        """Multi-scale variant of color-weighted matching.

        Tries the template at each scale in scale_range, runs 3-channel
        matching at each scale, and returns the best overall result.
        """
        th, tw = template.shape[:2]
        best: dict | None = None

        for scale in scale_range:
            if scale == 1.0:
                tpl_scaled = template
            else:
                sw = max(6, int(tw * scale))
                sh = max(6, int(th * scale))
                if sw > screen.shape[1] or sh > screen.shape[0]:
                    continue
                tpl_scaled = cv2.resize(template, (sw, sh), interpolation=cv2.INTER_AREA)

            result = self._match_color_weighted(
                screen, tpl_scaled, template_name, threshold, method,
            )
            if result is not None:
                result["scale"] = scale
                if best is None or result["score"] > best["score"]:
                    best = result

        if best is not None:
            logger.debug(
                "Template '%s' color_weighted multi_scale best: score=%.4f, scale=%.2f",
                template_name, best["score"], best.get("scale", 1.0),
            )
        return best

    def find_all_matches(
        self,
        screenshot: Image.Image | str | Path,
        template_name: str,
        threshold: float | None = None,
        min_distance: int = 30,
        grayscale: bool = False,
    ) -> list[dict]:
        """Find ALL template matches above threshold, not just the best one.

        Uses non-maximum suppression to avoid duplicate detections of the
        same badge. Each returned match is guaranteed to be at least
        min_distance pixels away from all other matches.

        Args:
            screenshot: PIL image, path, or ndarray.
            template_name: Name of the loaded template.
            threshold: Minimum confidence score (0-1). Default from config.
            min_distance: Minimum pixel distance between match centers for
                          non-max suppression. Prevents duplicate hits.
            grayscale: If True, convert to grayscale before matching to
                       eliminate background-color variation.

        Returns:
            List of {name, score, center, bbox} dicts, sorted by score desc.
        """
        threshold = threshold or config.vision.template_match_threshold
        with self._lock:
            template = self._templates.get(template_name)
        if template is None:
            logger.warning("Template '%s' not loaded", template_name)
            return []
        screen = self._to_cv(screenshot)

        if grayscale:
            if len(template.shape) == 3:
                template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            if len(screen.shape) == 3:
                screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

        result = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
        h, w = template.shape[:2]

        # Find all pixel locations above threshold
        locations = np.where(result >= threshold)
        # Convert to list of (score, x, y)
        candidates = [(result[y, x], x, y) for y, x in zip(*locations)]

        if not candidates:
            return []

        # Sort by score descending
        candidates.sort(key=lambda c: c[0], reverse=True)

        # Non-maximum suppression
        matches: list[dict] = []
        for score, px, py in candidates:
            cx = px + w // 2
            cy = py + h // 2
            # Check distance to all kept matches
            too_close = False
            for m in matches:
                dx = m["center"][0] - cx
                dy = m["center"][1] - cy
                if (dx * dx + dy * dy) < (min_distance * min_distance):
                    too_close = True
                    break
            if too_close:
                continue
            matches.append({
                "name": template_name,
                "score": float(score),
                "center": (cx, cy),
                "bbox": (px, py, px + w, py + h),
            })

        logger.debug(
            "find_all_matches: '%s' — %d candidates, %d after NMS (th=%.2f)",
            template_name, len(candidates), len(matches), threshold,
        )
        return matches

    def cache_position(self, template_name: str, x: int, y: int) -> None:
        """Store a position with timestamp. Evicts oldest if over limit."""
        with self._lock:
            now = time.monotonic()
            # Touch: move to end if already present (LRU semantics)
            if template_name in self._position_cache:
                del self._position_cache[template_name]
            self._position_cache[template_name] = (x, y, now)
            # Evict oldest entries if over capacity
            while len(self._position_cache) > self._max_cache_entries:
                self._position_cache.popitem(last=False)

    def get_cached_position(self, template_name: str) -> tuple[int, int] | None:
        """Return cached (x, y) if entry exists and hasn't expired."""
        with self._lock:
            entry = self._position_cache.get(template_name)
            if entry is None:
                return None
            x, y, ts = entry
            if time.monotonic() - ts > self._cache_ttl:
                del self._position_cache[template_name]
                return None
            # Touch: move to end to refresh LRU position
            del self._position_cache[template_name]
            self._position_cache[template_name] = (x, y, ts)
            return (x, y)

    def clear_position_cache(self) -> None:
        with self._lock:
            self._position_cache.clear()

    @property
    def loaded_templates(self) -> list[str]:
        with self._lock:
            return list(self._templates.keys())

    @staticmethod
    def _to_cv(image: Image.Image | str | Path) -> np.ndarray:
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        arr = np.array(image.convert("RGB"))
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


template_matcher = TemplateMatcher()
