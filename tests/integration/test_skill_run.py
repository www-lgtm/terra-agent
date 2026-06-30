"""Integration test: execute a skill end-to-end.

Requires a connected emulator with Arknights running.
"""

import pytest
from src.agent.loop import TerraAgent
from src.device.emulator import emulator_manager
from src.device.adb import init_adb


@pytest.mark.slow
def test_run_simple_task():
    serial = emulator_manager.first_online
    if not serial:
        pytest.skip("No ADB device available")

    init_adb(serial)
    agent = TerraAgent(device_serial=serial)
    result = agent.run("截图看看现在是什么画面")
    print(f"Result: {result}")
    assert "error" not in result or result.get("final_response"), "Should return a response"
