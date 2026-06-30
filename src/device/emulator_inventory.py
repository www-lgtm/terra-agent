"""EmulatorInventory — persistent emulator registry with game tracking.

Stores under data/emulators/inventory.json. Agent and user co-manage
this through WeChat conversation.

Schema per entry:
    {
      "id": "mumu_main",
      "name": "MuMu 12 主模拟器",
      "emulator_type": "mumu",
      "console_path": "D:\\...\\MuMuPlayer.exe",
      "installed_games": ["arknights", "reverse1999"],
      "aliases": ["MuMu", "主号"],
      "adb_ports": ["16384", "7555", "7556"],
    }

current_serial is runtime-only — tracked by EmulatorManager, not persisted.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config.settings import config

logger = logging.getLogger(__name__)

_INVENTORY_PATH = Path(config.DATA_DIR) / "emulators" / "inventory.json"


@dataclass
class EmulatorEntry:
    """One registered emulator in the inventory."""

    id: str                          # "mumu_main", "ldplayer_jp"
    name: str                        # "MuMu 12 主模拟器"
    emulator_type: str               # "mumu" | "ldplayer"
    console_path: str                # path to exe
    installed_games: list[str]       # ["arknights", "reverse1999"]
    aliases: list[str]               # ["MuMu", "主号"]
    adb_ports: list[str]             # ["16384", "7555"]

    # Runtime — set by EmulatorManager, NOT persisted
    current_serial: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "emulator_type": self.emulator_type,
            "console_path": self.console_path,
            "installed_games": self.installed_games,
            "aliases": self.aliases,
            "adb_ports": self.adb_ports,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EmulatorEntry:
        return cls(
            id=d["id"],
            name=d["name"],
            emulator_type=d["emulator_type"],
            console_path=d.get("console_path", ""),
            installed_games=d.get("installed_games", []),
            aliases=d.get("aliases", []),
            adb_ports=d.get("adb_ports", []),
        )


class EmulatorInventory:
    """Persistent registry of emulators and their installed games.

    Thread-safe.  Persists to data/emulators/inventory.json.
    """

    def __init__(self) -> None:
        self._entries: dict[str, EmulatorEntry] = {}
        self._lock = threading.Lock()
        self._loaded = False

    # ── Persistence ────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            self._load()
            self._loaded = True

    def _load(self) -> None:
        p = _INVENTORY_PATH
        if not p.exists():
            logger.info("No emulator inventory found at %s — creating default", p)
            self._create_default()
            return
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            entries_list = data.get("emulators", data if isinstance(data, list) else [])
            if isinstance(data, dict) and "emulators" in data:
                entries_list = data["emulators"]
            self._entries = {}
            for d in entries_list:
                entry = EmulatorEntry.from_dict(d)
                self._entries[entry.id] = entry
            logger.info("Loaded %d emulator(s) from inventory", len(self._entries))
        except Exception:
            logger.warning("Failed to load emulator inventory — creating default", exc_info=True)
            self._create_default()

    def _create_default(self) -> None:
        """Create a default inventory from current config.emulator settings."""
        emu_type = config.emulator.type
        instance = config.emulator.instance_name or ""
        console = config.emulator.console_path or ""
        from src.games.registry import get_game_registry
        all_games = get_game_registry().get_ids()

        default_ports = _DEFAULT_ADB_PORTS.get(emu_type, ["5555"])
        entry = EmulatorEntry(
            id="default",
            name=f"{emu_type} 模拟器" + (f" ({instance})" if instance else ""),
            emulator_type=emu_type,
            console_path=console,
            installed_games=list(all_games),  # Assume all games installed initially
            aliases=["主模拟器", emu_type],
            adb_ports=default_ports,
        )
        self._entries = {entry.id: entry}
        p = _INVENTORY_PATH
        p.parent.mkdir(parents=True, exist_ok=True)
        self._save_locked()
        logger.info("Created default emulator inventory with %d game(s)", len(all_games))

    def _save_locked(self) -> None:
        """Caller must hold _lock."""
        p = _INVENTORY_PATH
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "emulators": [e.to_dict() for e in self._entries.values()],
        }
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def save(self) -> None:
        self._ensure_loaded()
        with self._lock:
            self._save_locked()

    # ── Query ───────────────────────────────────────────────────────

    def list_all(self) -> list[EmulatorEntry]:
        self._ensure_loaded()
        with self._lock:
            return list(self._entries.values())

    def get(self, emu_id: str) -> EmulatorEntry | None:
        self._ensure_loaded()
        with self._lock:
            return self._entries.get(emu_id)

    def find_by_alias(self, text: str) -> EmulatorEntry | None:
        """Find an emulator by name or alias substring match."""
        self._ensure_loaded()
        text_lower = text.lower()
        with self._lock:
            for e in self._entries.values():
                if e.name.lower() in text_lower:
                    return e
                for a in e.aliases:
                    if a.lower() in text_lower:
                        return e
            return None

    def find_for_game(self, game_id: str) -> list[EmulatorEntry]:
        """Return all emulators that have this game installed."""
        self._ensure_loaded()
        with self._lock:
            return [e for e in self._entries.values()
                   if game_id in e.installed_games]

    def find_by_serial(self, serial: str) -> EmulatorEntry | None:
        """Find which emulator this ADB serial currently belongs to."""
        self._ensure_loaded()
        with self._lock:
            for e in self._entries.values():
                if e.current_serial == serial:
                    return e
            return None

    def update_serial(self, emu_id: str, serial: str) -> None:
        self._ensure_loaded()
        with self._lock:
            e = self._entries.get(emu_id)
            if e:
                e.current_serial = serial

    def all_known_ports(self) -> list[str]:
        """All ADB ports across all registered emulators."""
        self._ensure_loaded()
        with self._lock:
            ports: list[str] = []
            for e in self._entries.values():
                for p in e.adb_ports:
                    if p not in ports:
                        ports.append(p)
            return ports

    def all_online_serials(self, emu_manager: Any) -> dict[str, str]:
        """Map emulator_id → serial for online devices only.

        Uses emu_manager.discover() to get raw ADB state, then matches
        serials back to inventory entries by port pattern.
        """
        self._ensure_loaded()
        emu_manager.discover()
        all_online = [s for s, st in emu_manager._devices.items() if st == "device"]
        result: dict[str, str] = {}
        with self._lock:
            for e in self._entries.values():
                for s in all_online:
                    # Match by port — e.g. entry has ports ["16384"], serial is "127.0.0.1:16384"
                    for port in e.adb_ports:
                        if f":{port}" in s or s.endswith(f":{port}"):
                            result[e.id] = s
                            e.current_serial = s
                            break
                    if e.id in result:
                        break
        return result

    # ── Auto-discovery ─────────────────────────────────────────────

    @staticmethod
    def _build_game_package_map() -> dict[str, str]:
        """Build {package_name: game_id} from all registered GamePlugins."""
        mapping: dict[str, str] = {}
        from src.games.registry import get_game_registry
        for plugin in get_game_registry().list_all():
            for pkg in plugin.manifest.android_packages:
                mapping[pkg] = plugin.manifest.id
        return mapping

    def auto_discover_device(self, serial: str,
                              adb_path: str = "adb") -> dict[str, str] | None:
        """Discover installed games on a device via adb shell pm list packages.

        Returns {game_id: package_name, ...} of detected games, or None on failure.
        Side effect: updates the matching inventory entry's installed_games.
        """
        import subprocess as _sp
        pkg_map = self._build_game_package_map()
        if not pkg_map:
            logger.debug("No game packages registered — skipping auto-discovery")
            return {}

        try:
            proc = _sp.run(
                [adb_path, "-s", serial, "shell", "pm", "list", "packages", "-3"],
                capture_output=True, text=True, timeout=15.0,
            )
        except Exception as e:
            logger.warning("pm list packages failed for %s: %s", serial, e)
            return None

        installed: dict[str, str] = {}
        for line in proc.stdout.strip().split("\n"):
            line = line.strip()
            if not line.startswith("package:"):
                continue
            pkg = line[8:]  # strip "package:" prefix
            if pkg in pkg_map:
                game_id = pkg_map[pkg]
                installed[game_id] = pkg
                logger.info("auto-discover: %s has %s (%s)", serial, game_id, pkg)

        # Update inventory entry
        if installed:
            port = serial.split(":")[-1] if ":" in serial else serial
            self._ensure_loaded()
            with self._lock:
                for e in self._entries.values():
                    if port in e.adb_ports:
                        old_games = set(e.installed_games)
                        new_games = set(installed.keys())
                        e.installed_games = list(new_games | old_games)
                        added = new_games - old_games
                        if added:
                            self._save_locked()
                            logger.info("Inventory: %s added games %s", e.name, added)
                        break

        return installed

    def auto_discover_all(self, online_serials: list[str],
                           adb_path: str = "adb") -> dict[str, dict[str, str]]:
        """Run auto-discovery on all online devices.

        Returns {serial: {game_id: package_name}} for all devices.
        Also registers new devices not yet in inventory.
        """
        result: dict[str, dict[str, str]] = {}
        for serial in online_serials:
            found = self.auto_discover_device(serial, adb_path=adb_path)
            if found is not None:
                result[serial] = found

        # Register devices that aren't in any inventory entry
        self._ensure_loaded()
        with self._lock:
            known_ports: set[str] = set()
            for e in self._entries.values():
                known_ports.update(e.adb_ports)
            for serial in result:
                port = serial.split(":")[-1] if ":" in serial else serial
                if port not in known_ports:
                    # New device — create a minimal entry
                    emu_type = config.emulator.type
                    entry = EmulatorEntry(
                        id=f"auto_{port}",
                        name=f"未知模拟器 ({port})",
                        emulator_type=emu_type,
                        console_path="",
                        installed_games=list(result[serial].keys()),
                        aliases=[],
                        adb_ports=[port],
                    )
                    self._entries[entry.id] = entry
                    self._save_locked()
                    logger.info("Auto-registered new emulator: %s (%s)", entry.name, port)

        return result

    # ── Mutate ──────────────────────────────────────────────────────

    def register_or_update(self, entry: EmulatorEntry) -> None:
        """Add or update an emulator entry. Persists immediately."""
        self._ensure_loaded()
        with self._lock:
            self._entries[entry.id] = entry
            self._save_locked()
        logger.info("Emulator inventory updated: %s (%s)", entry.id, entry.name)

    def add_game(self, emu_id: str, game_id: str) -> bool:
        """Record that a game is installed on an emulator. Returns False if not found."""
        self._ensure_loaded()
        with self._lock:
            e = self._entries.get(emu_id)
            if not e:
                return False
            if game_id not in e.installed_games:
                e.installed_games.append(game_id)
                self._save_locked()
                logger.info("Added game %s to emulator %s", game_id, emu_id)
            return True

    def remove_game(self, emu_id: str, game_id: str) -> bool:
        self._ensure_loaded()
        with self._lock:
            e = self._entries.get(emu_id)
            if not e:
                return False
            if game_id in e.installed_games:
                e.installed_games.remove(game_id)
                self._save_locked()
                logger.info("Removed game %s from emulator %s", game_id, emu_id)
            return True

    def remove(self, emu_id: str) -> bool:
        """Remove an emulator from inventory. Returns False if not found."""
        self._ensure_loaded()
        with self._lock:
            if emu_id not in self._entries:
                return False
            del self._entries[emu_id]
            self._save_locked()
            logger.info("Removed emulator %s from inventory", emu_id)
            return True


# ── Default ADB ports by emulator type ─────────────────────────────

_DEFAULT_ADB_PORTS: dict[str, list[str]] = {
    "mumu": ["16384", "7555", "7556", "7557", "7558"],
    "ldplayer": ["5555"],
    "bluestacks": ["5555"],
    "generic": ["5555"],
}


# ── Singleton ──────────────────────────────────────────────────────

_inventory: EmulatorInventory | None = None


def get_emulator_inventory() -> EmulatorInventory:
    global _inventory
    if _inventory is None:
        _inventory = EmulatorInventory()
    return _inventory


# ── Format helper for agent display ─────────────────────────────────

def format_inventory_for_agent(inventory: EmulatorInventory,
                               online_map: dict[str, str]) -> str:
    """Build a human-readable summary of the emulator landscape."""
    entries = inventory.list_all()
    if not entries:
        return "📱 当前没有注册任何模拟器。"
    from src.games.registry import get_game_registry
    gr = get_game_registry()
    lines = ["📱 **模拟器清单**\n"]
    for e in entries:
        serial = online_map.get(e.id, "")
        status = f"🟢 在线 ({serial})" if serial else "⚫ 离线"
        games = ", ".join(gr.get_game_name(g) for g in e.installed_games) if e.installed_games else "（未记录）"
        lines.append(
            f"**{e.name}** [{e.emulator_type}] — {status}\n"
            f"  已安装游戏: {games}\n"
            f"  ADB 端口: {', '.join(e.adb_ports)}"
        )
    return "\n".join(lines)
