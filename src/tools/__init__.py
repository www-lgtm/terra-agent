"""Tools layer: ADB control, vision, wait, safety, confidence, skill management.

Import this module to register all built-in tools with the registry.
"""

# Import tool modules to trigger self-registration
from src.tools import adb_control  # noqa: F401
from src.tools import ask_user  # noqa: F401
from src.tools import base_collect_tool  # noqa: F401
from src.tools import base_shift_tool  # noqa: F401
from src.tools import box_scan_tool  # noqa: F401
from src.tools import confidence  # noqa: F401
from src.tools import depot_scan_tool  # noqa: F401
from src.tools import emulator_tools  # noqa: F401
from src.tools import knowledge_tool  # noqa: F401
from src.tools import material_tools  # noqa: F401

from src.tools import learn  # noqa: F401
from src.tools import notify_screen  # noqa: F401
from src.tools import recruit_optimizer  # noqa: F401
from src.tools import remember  # noqa: F401
from src.tools import safety  # noqa: F401
from src.tools import schedule_tools  # noqa: F401
from src.tools import scheduler_tool  # noqa: F401
from src.tools import skill_run  # noqa: F401
from src.tools import vision  # noqa: F401
from src.tools import wait_tool  # noqa: F401
from src.tools.registry import registry  # noqa: F401
