"""GameSlot and SlotRouter — message to device-pool routing.

核心原则：目标唯一 → 执行；目标不唯一 → 反问。

设备池设计：
- GameSlot 代表一个设备的占用状态，game 是该设备当前运行的游戏
- game 是运行时状态（任务执行后更新），不是固定绑定
- 路由时：优先匹配已在该游戏上的设备，空闲设备可以接任何游戏
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GameSlot:
    """一个设备槽位——路由和排队的原子单位。

    每个 GameSlot 代表一个模拟器设备。game 是该设备当前运行的游戏
    （运行时更新，空字符串 = 未绑定/游戏未知）。
    """

    slot_id: str                    # "dev_16384" / "dev_7555"
    label: str                      # 用户可见: "设备 16384" / "设备 7555"
    aliases: list[str]              # 别名: ["主号","大号"] / ["1999模拟器"]
    game: str = ""                  # 当前运行的游戏（运行时更新），""=未绑定
    device_serial: str = ""         # 底层 ADB 设备
    current_task: Any | None = None # AgentHandle | None
    pending_tasks: list[Any] = field(default_factory=list)

    @property
    def is_free(self) -> bool:
        if self.current_task is None:
            return True
        # AgentHandle: check if the underlying agent is running
        handle = self.current_task
        if hasattr(handle, 'is_running'):
            return not handle.is_running
        # Legacy: TerraAgent directly
        return not handle.state.running

    @property
    def status_text(self) -> str:
        if self.current_task is None:
            return "空闲"
        handle = self.current_task
        if hasattr(handle, 'status_text'):
            return handle.status_text
        # Legacy: TerraAgent directly
        agent = handle
        if agent.state.running:
            return agent.state.current_activity or agent.state.task_description or "执行中"
        return "空闲"

    def match_label(self, text: str) -> bool:
        """检查文本是否匹配此 slot 的 label 或任一 alias。"""
        candidates = [self.label] + self.aliases
        text_lower = text.lower()
        return any(c.lower() in text_lower for c in candidates)

    @property
    def game_label(self) -> str:
        if not self.game:
            return "（待分配）"
        from src.games.registry import get_game_registry
        plugin = get_game_registry().get(self.game)
        return plugin.manifest.name if plugin else self.game

    @classmethod
    def from_config(cls, cfg: dict) -> GameSlot:
        return cls(
            slot_id=cfg["slot_id"],
            label=cfg["label"],
            aliases=cfg.get("aliases", []),
            game=cfg.get("game", ""),
            device_serial=cfg["device_serial"],
        )


# ── Route result types ──


@dataclass
class RouteUnique:
    """目标唯一 — 可以直接执行"""
    slot: GameSlot


@dataclass
class RouteAmbiguous:
    """目标不唯一 — 必须反问用户"""
    candidates: list[GameSlot]
    reason: str  # "multiple_slots" | "no_game_specified"
    user_task: str = ""


@dataclass
class RouteNone:
    """没有匹配的 slot"""
    message: str


@dataclass
class RouteBatch:
    """批量路由 — 一个任务发给多个 slot"""
    slots: list[GameSlot]
    task: str  # 去掉批量关键词后的纯任务描述


# ── Batch keyword detection ──────────────────────────────────────

_BATCH_KW = ["两个号都", "两个号", "所有号", "每个号", "每个",
             "全部都", "全部", "所有", "都", "一起", "同时"]

_GAME_BATCH_KW = ["都"]

_ALL_KW = ["都", "全部", "所有", "每个", "所有号", "每个号", "全部都"]


def _extract_batch_task(text: str) -> str | None:
    """Strip batch keywords from text, returning the pure task description.

    Examples:
        "两个号都清体力" → "清体力"
        "方舟都基建收菜" → "基建收菜"
        "所有号刷GT-6" → "刷GT-6"
        "全部清体力" → "清体力"
    Returns None if text is not a batch command.
    """
    import re
    hit = any(kw in text for kw in _BATCH_KW)
    if not hit:
        return None

    cleaned = text
    # Strip all batch keywords (longest first to avoid partial matches)
    for kw in sorted(_BATCH_KW, key=len, reverse=True):
        cleaned = cleaned.replace(kw, "")
    cleaned = cleaned.strip()
    # Remove leading/trailing game names
    cleaned = re.sub(r'^(方舟|arknights|1999|重返未来)\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip()
    return cleaned if cleaned else None


# ── SlotRouter ──


class SlotRouter:
    """根据用户消息匹配设备槽位（设备池路由）。

    匹配优先级:
    1. 消息包含显式账号名(label/alias) → 直接匹配
    2. 消息包含游戏名 → 优先匹配已在该游戏上的设备
    3. 有空闲设备 → 分配空闲设备（agent 会自动切换游戏）
    4. 所有设备忙 → 反问用户
    """

    def __init__(self, slots: list[GameSlot], active_game: str = "arknights",
                 validate: bool = True) -> None:
        self._slots = slots
        self.active_game = active_game

        # 校验: slot.label 和 aliases 全局唯一
        if validate:
            seen: set[str] = set()
            for s in slots:
                for name in [s.label] + s.aliases:
                    lower = name.lower()
                    if lower in seen:
                        raise ValueError(
                            f"Duplicate slot label/alias: '{name}' — "
                            f"labels and aliases must be unique across all slots"
                        )
                    seen.add(lower)

    def set_active_game(self, game: str) -> None:
        self.active_game = game

    def route(self, text: str) -> RouteUnique | RouteAmbiguous | RouteNone:
        """匹配用户消息到设备槽位。"""
        text_lower = text.lower()

        # ── Step 1: 显式账号名 → 直接匹配 ──
        for s in self._slots:
            if s.match_label(text):
                return RouteUnique(slot=s)

        # ── Step 2: 检测消息中的游戏名 ──
        detected_game = self._detect_game_in_text(text_lower)

        # ── Step 3: 设备池分配 ──
        if detected_game:
            # 3a: 正在该游戏上的空闲设备 → 首选
            on_game_idle = [s for s in self._slots
                           if s.game == detected_game and s.is_free]
            if on_game_idle:
                return RouteUnique(slot=on_game_idle[0])

            # 3b: 该游戏上有设备但忙 + 有空闲设备 → 用空闲的（切换游戏）
            on_game = [s for s in self._slots if s.game == detected_game]
            any_idle = [s for s in self._slots if s.is_free]
            if on_game and any_idle:
                return RouteUnique(slot=any_idle[0])

            # 3c: 该游戏上有设备但忙 + 无空闲 → 排队
            if on_game:
                return RouteUnique(slot=on_game[0])

            # 3d: 没有设备在该游戏上 → 分配空闲设备
            if any_idle:
                return RouteUnique(slot=any_idle[0])

            # 3e: 全忙 → 反问
            if self._slots:
                return RouteAmbiguous(
                    candidates=self._slots,
                    reason="all_busy",
                    user_task=text,
                )
            return RouteNone(message="没有配置任何设备。")

        # ── Step 4: 没有显式游戏名 → 空闲设备优先，然后是 active_game 的设备 ──
        idle = [s for s in self._slots if s.is_free]
        if idle:
            # Prefer idle device already on active_game
            on_active_idle = [s for s in idle if s.game == self.active_game]
            return RouteUnique(slot=on_active_idle[0] if on_active_idle else idle[0])

        on_active = [s for s in self._slots if s.game == self.active_game]
        if on_active:
            return RouteUnique(slot=on_active[0])

        # ── Step 5: 全忙 → 反问 ──
        if self._slots:
            return RouteAmbiguous(
                candidates=self._slots,
                reason="all_busy",
                user_task=text,
            )
        return RouteNone(message="没有配置任何设备。")

    def _detect_game_in_text(self, text_lower: str) -> str | None:
        """检测文本中的游戏名。

        返回明确匹配的游戏（至少命中一个关键词），不返回默认值。
        """
        from src.games.registry import get_game_registry
        registry = get_game_registry()
        # Collect all keyword hits per game
        scores: dict[str, int] = {}
        for plugin in registry.list_all():
            for kw in plugin.manifest.keywords:
                if kw.lower() in text_lower:
                    scores[plugin.manifest.id] = scores.get(plugin.manifest.id, 0) + 1
        if scores:
            # Return the game with the most keyword hits
            return max(scores, key=scores.get)
        return None

    def list_games(self) -> list[str]:
        """返回所有已注册的游戏（设备池模式下 slot.game 可能为空）。"""
        games: list[str] = [s.game for s in self._slots if s.game]
        if not games:
            from src.games.registry import get_game_registry
            games = get_game_registry().get_ids()
        return games

    def detect_batch(self, text: str) -> RouteBatch | None:
        """检测批量任务意图。"""
        if not self._slots or len(self._slots) <= 1:
            return None

        task = _extract_batch_task(text)
        if task is None:
            return None

        # Step 1: explicit slot names (e.g. "主号小号都清")
        matched_slots: list[GameSlot] = []
        for s in self._slots:
            if s.match_label(text):
                matched_slots.append(s)
        if len(matched_slots) >= 2:
            clean_task = task
            for s in matched_slots:
                for name in [s.label] + s.aliases:
                    clean_task = clean_task.replace(name, "")
            clean_task = clean_task.strip()
            return RouteBatch(slots=matched_slots, task=clean_task if clean_task else task)

        # Step 2: generic batch keywords → all idle slots
        if any(kw in text for kw in _ALL_KW):
            idle = [s for s in self._slots if s.is_free]
            if idle:
                return RouteBatch(slots=idle, task=task)

        return None

    @property
    def has_multiple_games(self) -> bool:
        return len(self.list_games()) > 1

    @property
    def has_multiple_slots(self) -> bool:
        return len(self._slots) > 1
