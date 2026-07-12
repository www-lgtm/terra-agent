"""Terra Agent configuration loaded from environment variables and config files."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


@dataclass
class LLMConfig:
    provider: str = "anthropic"
    model: str = "mimo-v2.5"
    base_url: str = "https://api.xiaomimimo.com/anthropic"
    api_key: str = field(default_factory=lambda: os.getenv("MIMO_API_KEY", ""))
    max_tokens: int = 4096
    context_length: int = 131072
    temperature: float = 0.3          # For tool-calling precision (was 0.0 — too deterministic, caused repetition loops)
    chat_temperature: float = 0.7     # For text replies (quick_reply, casual chat)

    @property
    def api_keys(self) -> list[str]:
        """Return all configured API keys (comma-separated in env).

        When multiple keys are configured, each TerraAgent gets a different key
        to bypass per-key concurrency limits on the MiMo API server.

        Accepts both ASCII commas (,) and Chinese full-width commas (，) to
        avoid silent failures when the user accidentally uses the IME comma.
        """
        raw = self.api_key.strip()
        if not raw:
            return []
        # Normalize: replace Chinese full-width commas with ASCII commas
        raw = raw.replace("，", ",")
        keys = [k.strip() for k in raw.split(",") if k.strip()]
        return keys or [raw]

    def get_api_key(self, index: int = 0) -> str:
        """Return the API key at the given index (wraps around)."""
        keys = self.api_keys
        if not keys:
            return ""
        return keys[index % len(keys)]

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)


@dataclass
class ADBConfig:
    path: str = field(default_factory=lambda: os.getenv("ADB_PATH", "adb"))
    heartbeat_interval: float = 20.0
    max_reconnect_attempts: int = 5
    tap_noise_px: int = 5
    action_delay_min_ms: int = 200
    action_delay_max_ms: int = 500

    def __post_init__(self) -> None:
        if self.action_delay_min_ms > self.action_delay_max_ms:
            raise ValueError(
                f"action_delay_min_ms ({self.action_delay_min_ms}) must be <= "
                f"action_delay_max_ms ({self.action_delay_max_ms})"
            )
        if self.heartbeat_interval < 1:
            raise ValueError(f"heartbeat_interval must be >= 1, got {self.heartbeat_interval}")
        if self.tap_noise_px < 0:
            raise ValueError(f"tap_noise_px must be >= 0, got {self.tap_noise_px}")


@dataclass
class MAAToolsConfig:
    """MAA (MaaAssistantArknights) integration paths.

    MAA provides operator box scanning, depot item matching, and base-shift
    scheduling via its resource templates.  These tools are optional — set
    the path to enable them, or leave empty to skip MAA-dependent tools.
    """
    # Root directory of MAA installation or resource checkout.
    # Example: r"D:\MAA-v6.11.1-win-x64" or r"D:\vsworkspace\MaaAssistantArknights"
    root: str = field(default_factory=lambda: os.getenv("MAA_ROOT", ""))
    # Alternative: path to MAA resource/template directory directly.
    resource_dir: str = field(default_factory=lambda: os.getenv("MAA_RESOURCE_DIR", ""))

    @property
    def is_configured(self) -> bool:
        return bool(self.root) or bool(self.resource_dir)


@dataclass
class VisionConfig:
    ocr_confidence_threshold: float = 0.8
    vlm_cache_enabled: bool = True
    ocr_region_cache_enabled: bool = True


@dataclass
class SafetyConfig:
    max_daily_actions: int = 500

    # ── Per-game overrides ──────────────────────────────────────────
    # These are game-agnostic fallbacks.  When GameRegistry is available,
    # the per-game manifest values take precedence.
    # Set these only if you need defaults for unregistered games.
    dangerous_keywords: list[str] = field(default_factory=list)
    safe_compound_terms: list[str] = field(default_factory=list)
    require_confirmation_keywords: list[str] = field(default_factory=list)

    def effective_dangerous_keywords(self, game: str = "arknights") -> list[str]:
        """Get dangerous keywords for a game, preferring GameRegistry if available."""
        try:
            from src.games.registry import get_game_registry
            kw = get_game_registry().get_dangerous_keywords(game_id=game)
            if kw:
                return kw
        except Exception:
            pass
        return self.dangerous_keywords

    def effective_safe_compound_terms(self, game: str = "arknights") -> list[str]:
        try:
            from src.games.registry import get_game_registry
            terms = get_game_registry().get_safe_compound_terms(game_id=game)
            if terms:
                return terms
        except Exception:
            pass
        return self.safe_compound_terms

    def effective_confirmation_keywords(self, game: str = "arknights") -> list[str]:
        try:
            from src.games.registry import get_game_registry
            kw = get_game_registry().get_confirmation_keywords(game_id=game)
            if kw:
                return kw
        except Exception:
            pass
        return self.require_confirmation_keywords


@dataclass
class AgentConfig:
    max_iterations: int = 200
    task_timeout_seconds: float = 1800.0  # 30 min wall-clock timeout per task
    tool_execution_mode: str = "sequential"
    vision_mode: str = "auto_inject"  # "auto_inject" | "vlm_legacy"
    screenshot_max_width: int = 960
    screenshot_quality: int = 65
    # Memory lifecycle (Phase 1)
    memory_stale_days: int = 30
    memory_stale_low_success_ratio: float = 0.2
    # Skill refinement (auto-generate skill scripts from agent actions)
    # DEFAULT OFF: produces garbage v2 variants that pollute the skill index.
    # Enable only when actively developing and reviewing generated skills.
    enable_skill_refinement: bool = False
    # Skill generation (auto-create new skill .md files from action chains)
    # DEFAULT OFF: generates garbage with bad names and coordinates from
    # loading screens/misclicks. Hand-author skills instead.
    enable_skill_generation: bool = False
    # Learning engine (Phase 1-3)
    learning_injection_attribution_window: int = 3  # Turns after injection to check for help
    learning_pattern_miner_interval: int = 10       # Tasks between pattern miner runs
    learning_low_help_harm_ratio: float = 0.5       # help/harm ratio below which memory is harmful

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {self.max_iterations}")
        if self.vision_mode not in ("auto_inject", "vlm_legacy"):
            raise ValueError(f"vision_mode must be 'auto_inject' or 'vlm_legacy', got {self.vision_mode!r}")
        if self.memory_stale_days < 1:
            raise ValueError(f"memory_stale_days must be >= 1, got {self.memory_stale_days}")
        if not 0.0 <= self.memory_stale_low_success_ratio <= 1.0:
            raise ValueError(f"memory_stale_low_success_ratio must be 0.0-1.0, got {self.memory_stale_low_success_ratio}")


@dataclass
class EmulatorConfig:
    """Emulator lifecycle management.

    All paths can be overridden via environment variables.
    """
    # Emulator type: "ldplayer", "mumu", "bluestacks", or "generic"
    type: str = field(default_factory=lambda: os.getenv("EMULATOR_TYPE", "ldplayer"))

    # Path to the emulator's CLI tool (ldconsole.exe, MuMuManager.exe, etc.)
    console_path: str = field(default_factory=lambda: os.getenv(
        "EMULATOR_CONSOLE",
        r"C:\Program Files\ldplayer\ldconsole.exe",
    ))

    # Instance name/index (LDPlayer uses names like "雷电模拟器", MuMu uses indices)
    instance_name: str = field(default_factory=lambda: os.getenv("EMULATOR_INSTANCE", "雷电模拟器"))

    # ---- Memory watchdog ----
    # Max total emulator process memory in MB before auto-restart. 0 = disabled.
    memory_limit_mb: int = field(default_factory=lambda: int(os.getenv("EMULATOR_MEMORY_LIMIT_MB", "6144")))

    # Minimum interval between auto-restarts (seconds) — prevents restart storms
    restart_cooldown_seconds: int = 3600

    # ---- Scheduled restart ----
    # Cron expression for daily emulator restart. Empty string = disabled.
    restart_cron: str = field(default_factory=lambda: os.getenv("EMULATOR_RESTART_CRON", "0 4 * * *"))

    # ---- Restart timing ----
    # Seconds to wait for emulator to fully close
    shutdown_wait_seconds: float = 15.0
    # Seconds to wait for emulator to boot + ADB to become available
    boot_wait_seconds: float = 60.0
    # Seconds between ADB availability checks during boot
    adb_poll_interval: float = 3.0

    # ---- Process name patterns for memory monitoring ----
    # Process names to watch (case-insensitive match).  Empty = auto-detect by type.
    watch_process_names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        allowed = {"ldplayer", "mumu", "bluestacks", "generic"}
        if self.type not in allowed:
            raise ValueError(f"emulator.type must be one of {allowed}, got {self.type!r}")
        if self.restart_cooldown_seconds < 60:
            raise ValueError("restart_cooldown_seconds must be >= 60")


@dataclass
class SchedulerConfig:
    poll_interval: float = 30.0           # Seconds between polling for due tasks
    max_concurrent_per_device: int = 1    # Max tasks per device (fixed at 1)
    history_retention_days: int = 30      # Auto-delete schedule_history older than N days


@dataclass
class GatewayConfig:
    """网关（WeChat iLink）通信配置."""
    ssl_verify: bool = field(
        default_factory=lambda: os.getenv("GATEWAY_SSL_VERIFY", "true").lower() != "false"
    )
    ssl_cert_path: str = field(default_factory=lambda: os.getenv("GATEWAY_SSL_CERT_PATH", ""))
    token_encryption_key: str = field(default_factory=lambda: os.getenv("TOKEN_ENCRYPTION_KEY", ""))

    def __post_init__(self) -> None:
        if self.ssl_cert_path and not Path(self.ssl_cert_path).exists():
            raise ValueError(f"ssl_cert_path does not exist: {self.ssl_cert_path}")

    @property
    def ssl_verify_enabled(self) -> bool:
        """Only return True if explicitly configured AND cert path is valid."""
        if not self.ssl_verify:
            return False
        if self.ssl_cert_path:
            return Path(self.ssl_cert_path).exists()
        # Default: use system CA bundle
        return True


@dataclass
class ConciergeConfig:
    """管家智能体（Concierge Agent）配置 — 纯确定性路由，无 LLM。"""
    max_iterations: int = 5               # 保留兼容


@dataclass
class ObservationConfig:
    """观察学习系统配置 — 记录用户手动操作，生成 guide 型 skill."""
    poll_interval_ms: float = 1500        # 截图间隔 (ms)
    max_duration_s: int = 1800            # 最长记录时间 (秒，默认30分钟)
    dhash_change_threshold: int = 8       # dHash Hamming 距离阈值 (≥此值=显著变化)
    frame_max_width: int = 800            # 保存帧 JPEG 最大宽度 (px)
    frame_jpeg_quality: int = 50          # 保存帧 JPEG 质量 (1-100)
    significant_frames_min: int = 3       # 少于 N 个关键帧 → 跳过提取
    enable_vlm_description: bool = False  # Phase 2: VLM 描述关键帧

    def __post_init__(self) -> None:
        if self.poll_interval_ms < 200:
            raise ValueError(f"poll_interval_ms must be >= 200, got {self.poll_interval_ms}")
        if self.max_duration_s < 10:
            raise ValueError(f"max_duration_s must be >= 10, got {self.max_duration_s}")
        if self.dhash_change_threshold < 1 or self.dhash_change_threshold > 64:
            raise ValueError(f"dhash_change_threshold must be 1-64, got {self.dhash_change_threshold}")


@dataclass
class SessionDefaults:
    """Per-session defaults — lightweight config, NOT the runtime AgentState.

    The real runtime state is at src.agent.state.AgentState.
    This dataclass only holds config-level session bootstrapping defaults.
    """
    game: str = "arknights"
    device_serial: str = ""
    running: bool = False
    task_queue: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Config:
    DATA_DIR: Path = DATA_DIR
    PROJECT_ROOT: Path = PROJECT_ROOT
    llm: LLMConfig = field(default_factory=LLMConfig)
    adb: ADBConfig = field(default_factory=ADBConfig)
    emulator: EmulatorConfig = field(default_factory=EmulatorConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    gateway: GatewayConfig = field(default_factory=GatewayConfig)
    concierge: ConciergeConfig = field(default_factory=ConciergeConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    maa: MAAToolsConfig = field(default_factory=MAAToolsConfig)
    state: SessionDefaults = field(default_factory=SessionDefaults)

    # ── Phase 2: Multi-game / multi-account slots ──
    # Each slot maps a device serial to a game+account label.
    # Auto-detected from env or left empty for single-device auto-create.
    # Example (set via .env or override in code):
    #   GAME_SLOTS = [
    #     {"slot_id": "ark_main", "label": "方舟主号", "game": "arknights",
    #      "device_serial": "emulator-5554", "aliases": ["主号", "大号"]},
    #   ]
    game_slots: list[dict] = field(default_factory=list)


config = Config()
