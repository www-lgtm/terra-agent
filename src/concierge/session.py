"""管家会话状态数据类 — 每个 WeChat 用户一个实例。"""

from __future__ import annotations

import json
import logging
import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Sessions expire after 7 days of inactivity
_SESSION_TTL_SECONDS = 7 * 24 * 3600


@dataclass
class ConciergeSession:
    """单个用户的管家会话状态。

    每个微信用户对应一个 ConciergeSession。管家维持持久对话上下文，
    同时管理最多一个正在运行的游戏智能体。
    """

    user_id: str = ""
    conversation_history: list[dict[str, Any]] = field(default_factory=list)
    current_agent: Any | None = None          # TerraAgent | None — 当前唯一的游戏智能体
    agent_device_serial: str = ""             # 当前智能体占用的设备序列号
    pending_clarification: str | None = None  # 上次向用户追问的内容标记
    pending_task: str | None = None           # 触发追问的原始任务描述
    _last_clarification_type: str = ""        # clarification type before clearing (for re-routing)
    _agent_id_counter: int = 1
    _device_owned: bool = False               # 是否持有设备信号量

    # Game context — cross-message active_game persistence
    # Set from MessageRouter.__init__ after construction.
    game_ctx: Any | None = None

    def reset(self) -> None:
        """清空管家对话历史（保留智能体引用和设备信息）。"""
        self.conversation_history.clear()
        self.pending_clarification = None
        self.pending_task = None

    # ── 跨天会话持久化 ──────────────────────────────────────────

    def save_snapshot(self, data_dir: str = "") -> None:
        """Persist session summary to data/session/{user_id}.json.

        Stores: active_game, pending_clarification, pending_task,
        last N conversation turns as text-only summary (no screenshots).
        """
        try:
            from config.settings import DATA_DIR
            dir_path = Path(data_dir) if data_dir else DATA_DIR / "session"
            dir_path.mkdir(parents=True, exist_ok=True)
            filepath = dir_path / f"{self.user_id}.json"

            # Summarize recent conversation: last 10 turns, text-only, max 500 chars each
            summary_turns: list[str] = []
            for msg in self.conversation_history[-10:]:
                content = msg.get("content", "")
                if isinstance(content, str):
                    summary_turns.append(content[:500])
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            summary_turns.append(block["text"][:500])

            data = {
                "user_id": self.user_id,
                "active_game": self.game_ctx.active_game if self.game_ctx else "unknown",
                "pending_clarification": self.pending_clarification,
                "pending_task": self.pending_task,
                "conversation_turns": summary_turns,
                "updated_at": _time.time(),
            }
            filepath.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            logger.debug("Session snapshot saved for %s (%d turns)",
                        self.user_id[:20], len(summary_turns))
        except Exception as e:
            logger.warning("Failed to save session snapshot for %s: %s",
                         self.user_id[:20], e)

    def load_snapshot(self, data_dir: str = "") -> bool:
        """Restore session from data/session/{user_id}.json.

        Returns True if a valid snapshot was loaded, False otherwise.
        Expired snapshots (>7 days) are treated as not found.
        """
        try:
            from config.settings import DATA_DIR
            dir_path = Path(data_dir) if data_dir else DATA_DIR / "session"
            filepath = dir_path / f"{self.user_id}.json"
            if not filepath.exists():
                return False

            data = json.loads(filepath.read_text(encoding="utf-8"))

            # Check expiry
            updated_at = data.get("updated_at", 0)
            if _time.time() - updated_at > _SESSION_TTL_SECONDS:
                logger.debug("Session snapshot for %s expired (%.0fh old), ignoring",
                           self.user_id[:20], (_time.time() - updated_at) / 3600)
                try:
                    filepath.unlink()
                except Exception:
                    pass
                return False

            # Restore state
            self.pending_clarification = data.get("pending_clarification")
            self.pending_task = data.get("pending_task")

            # Restore active_game to game_ctx
            active_game = data.get("active_game", "")
            if active_game and active_game != "unknown" and self.game_ctx is not None:
                self.game_ctx.active_game = active_game

            # Inject conversation summary as a system message so the
            # concierge can see recent context on the first message
            turns = data.get("conversation_turns", [])
            if turns:
                self.conversation_history = [{
                    "role": "user",
                    "content": (
                        "[系统提示 — 上次会话摘要] 以下是重启前最近的对话记录：\n"
                        + "\n".join(f"· {t}" for t in turns[-5:])
                    ),
                }]
                logger.debug("Session snapshot restored for %s (%d turns, game=%s)",
                           self.user_id[:20], len(turns), active_game)
            else:
                logger.debug("Session snapshot restored for %s (no conversation history)",
                           self.user_id[:20])

            return True
        except Exception as e:
            logger.warning("Failed to load session snapshot for %s: %s",
                         self.user_id[:20], e)
            return False

    # ── 截图缓存 (层级 3, Phase 3) ──
    # TODO: When user asks "刚才那个界面是什么", cache last 5 screenshots
    # (dhash → b64), do VLM describe, reply with text.  Screenshots are NOT
    # in LLM context — only looked up on demand.

    def cache_screenshot(self, dhash: str, b64: str) -> None:
        """TODO Phase 3: Store screenshot in LRU cache."""
        pass

    def lookup_screenshot(self, dhash: str) -> str | None:
        """TODO Phase 3: Retrieve cached screenshot b64 by dhash."""
        return None
