"""CLI entry point for Terra Agent.

Usage:
    python -m src.cli chat          # Interactive chat mode
    python -m src.cli run <task>    # Run a single task
    python -m src.cli skills        # List skills
    python -m src.cli status        # Show device and config status
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def setup_logging(verbose: bool = False) -> None:
    from config.settings import config as app_config
    from logging.handlers import RotatingFileHandler
    import time
    level = logging.DEBUG if verbose else logging.INFO
    log_dir = Path(app_config.DATA_DIR) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.StreamHandler(),
        RotatingFileHandler(
            log_dir / "terra_agent.log",
            encoding="utf-8",
            maxBytes=10 * 1024 * 1024,   # 10 MB per file
            backupCount=5,                 # Keep 5 rotated files
        ),
    ]
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s]%(agent_tag)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )
    # Inject per-agent tag into log records (thread-local, set by TerraAgent)
    from src.utils.log_tags import AgentTagFilter
    root_logger = logging.getLogger()
    for h in root_logger.handlers:
        h.addFilter(AgentTagFilter())


def setup_container() -> None:
    """Initialize the DI container. Must be called after setup_logging()
    and before any business logic that uses container services."""
    from src.container import get_container
    get_container()


def cmd_chat(args: list[str]) -> None:
    """Interactive CLI chat with the agent."""
    from rich.console import Console
    from rich.panel import Panel

    console = Console()

    # Check for device
    from src.device.emulator import emulator_manager
    serial = emulator_manager.first_online
    if not serial:
        console.print("[red]No ADB device found. Please connect a device or emulator first.[/red]")
        console.print("[yellow]You can still chat, but ADB tools will be unavailable.[/yellow]")
        serial = "emulator-5554"  # Default for development

    try:
        from src.device.adb import init_adb
        adb = init_adb(serial)
        console.print(f"[green]Connected to device: {serial}[/green]")
    except Exception as e:
        console.print(f"[yellow]ADB warning: {e}[/yellow]")
        console.print("[yellow]Continuing in offline mode. ADB tools will fail.[/yellow]")

    # Check LLM
    from config.settings import config as app_config
    if not app_config.llm.is_configured:
        console.print("[red]MIMO_API_KEY not set. Please set it in .env file.[/red]")
        return

    from src.agent.loop import TerraAgent

    def ask_fn(question: str) -> str:
        console.print(f"\n[yellow]Agent asks:[/yellow] {question}")
        return console.input("[bold yellow]你的回答:[/bold yellow] ").strip()

    agent = TerraAgent(device_serial=serial, ask_fn=ask_fn)

    console.print(Panel("[bold cyan]Terra Agent[/bold cyan] — Arknights Game Assistant"))
    console.print("Type your commands (e.g. '清体力', '刷GT-6'). Type 'quit' to exit.\n")

    while True:
        try:
            user_input = console.input("[bold green]你:[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            console.print("Goodbye!")
            break

        if user_input.lower() == "skills":
            from src.tools.skill_run import skill_list
            console.print(skill_list())
            continue

        if user_input.lower() == "status":
            from src.tools.registry import registry
            names = registry.get_names()
            console.print(f"Device: {serial}")
            console.print(f"Available tools ({len(names)}): {', '.join(names)}")
            continue

        with console.status("[cyan]Thinking...[/cyan]"):
            result = agent.run(user_input)

        if result.get("success"):
            console.print(f"[cyan]Agent:[/cyan] {result.get('final_response', 'Done.')}")
            console.print(f"[dim]({result.get('iterations', 0)} iterations)[/dim]")
        else:
            console.print(f"[red]Error:[/red] {result.get('error', 'Unknown error')}")


def cmd_run(args: list[str]) -> None:
    """Run a single task."""
    from src.device.emulator import emulator_manager
    from src.device.adb import init_adb
    from src.agent.loop import TerraAgent

    task = " ".join(args)
    if not task:
        print("Usage: python -m src.cli run <task>")
        sys.exit(1)

    serial = emulator_manager.first_online
    if not serial:
        print("No ADB device found.")
        sys.exit(1)

    init_adb(serial)
    agent = TerraAgent(device_serial=serial)
    result = agent.run(task)

    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_skills(args: list[str]) -> None:
    """List available skills."""
    from src.skills.manager import get_skill_manager
    mgr = get_skill_manager()
    names = mgr.list_all()
    if not names:
        print("No skills found.")
        return
    print(f"Available skills ({len(names)}):")
    for name in names:
        skill = mgr.load(name)
        if skill:
            desc = skill.get("description", "")
            print(f"  {name}: {desc}")


def cmd_status(args: list[str]) -> None:
    """Show device and config status."""
    from src.device.emulator import emulator_manager
    from config.settings import config as app_config

    print(f"LLM: {app_config.llm.model} @ {app_config.llm.base_url}")
    print(f"API key configured: {app_config.llm.is_configured}")

    devices = emulator_manager.discover()
    print(f"ADB devices: {len(devices)}")
    for serial, state in devices:
        status_icon = "OK" if state == "device" else state
        print(f"  {serial}: {status_icon}")


def cmd_weixin(args: list[str]) -> None:
    """Start WeChat iLink Bot (QR login + agent loop via WeChat)."""
    import asyncio
    from src.gateway.weixin import run_bot

    project_root = Path(__file__).resolve().parent.parent
    data_home = str(project_root / "data")
    print(f"Data home: {data_home}")
    asyncio.run(run_bot(data_home))


def cmd_schedule(args: list[str]) -> None:
    """Manage scheduled tasks: list, add, remove, enable, disable, run, daemon.

    Usage:
        schedule list                          # List all scheduled tasks
        schedule add <name> --cron <expr> --task <desc> [--one-shot]
        schedule add <name> --interval <val> --task <desc> [--one-shot]
        schedule remove <id>                   # Delete a task
        schedule enable <id>                   # Enable a task
        schedule disable <id>                  # Disable (pause) a task
        schedule run <id>                      # Trigger a task immediately
        schedule daemon                        # Start the background scheduler
    """
    import time
    from config.settings import config
    from src.scheduler.schedule_db import schedule_db

    if not args:
        print("Usage: schedule [list|add|remove|enable|disable|run|daemon]")
        print("Try 'schedule list' to see all tasks.")
        return

    action = args[0]

    if action == "list":
        tasks = schedule_db.get_all()
        if not tasks:
            print("No scheduled tasks.")
            return
        print(f"{'ID':<5} {'Status':<8} {'Name':<20} {'Schedule':<22} {'Next Run':<20} {'Runs':<6}")
        print("-" * 85)
        for t in tasks:
            status = "ENABLED" if t["enabled"] else "DISABLED"
            next_run = time.strftime("%Y-%m-%d %H:%M", time.localtime(t["next_run"])) if t["next_run"] else "N/A"
            sched = f"{t['schedule_type']}={t['schedule_value']}"
            print(f"#{t['id']:<4} {status:<8} {t['name'][:20]:<20} {sched[:22]:<22} {next_run:<20} {t['run_count']:<6}")

    elif action == "add":
        name = args[1] if len(args) > 1 else ""
        schedule_type = ""
        schedule_value = ""
        task_desc = ""
        slot_id = ""
        one_shot = "--one-shot" in args

        i = 2
        while i < len(args):
            if args[i] == "--cron":
                schedule_type = "cron"
                i += 1
                if i < len(args):
                    schedule_value = args[i]
            elif args[i] == "--interval":
                schedule_type = "interval"
                i += 1
                if i < len(args):
                    schedule_value = args[i]
            elif args[i] == "--task":
                i += 1
                if i < len(args):
                    task_desc = args[i]
            elif args[i] == "--slot":
                i += 1
                if i < len(args):
                    slot_id = args[i]
            i += 1

        if not name or not schedule_type or not schedule_value or not task_desc:
            print("Usage: schedule add <name> --cron/--interval <value> --task <description> [--one-shot] [--slot <slot_id>]")
            print('Example: schedule add 早间清体力 --cron "0 9 * * *" --task "清体力用GT-6刷糖" --slot ark_main')
            print('Example: schedule add 定时收菜 --interval 30m --task "基建收菜" --one-shot')
            return

        try:
            task_id = schedule_db.create(
                name=name,
                task_payload={"custom_prompt": task_desc},
                schedule_type=schedule_type,
                schedule_value=schedule_value,
                description=task_desc,
                one_shot=one_shot,
                slot_id=slot_id,
            )
        except ValueError as e:
            print(f"Error: Invalid schedule value: {e}")
            return
        except Exception as e:
            print(f"Error creating schedule: {e}")
            return
        print(f"Created schedule #{task_id}: {name} ({schedule_type}={schedule_value})")

    elif action == "remove":
        if len(args) < 2:
            print("Usage: schedule remove <id>")
            return
        try:
            task_id = int(args[1])
        except ValueError:
            print(f"Invalid task ID: {args[1]}")
            return
        # Cancel running task before deleting
        from src.scheduler.cron_scheduler import get_engine
        engine = get_engine()
        engine.cancel_task(task_id)
        if schedule_db.delete(task_id):
            print(f"Deleted schedule #{task_id}.")
        else:
            print(f"Schedule #{task_id} not found.")

    elif action == "enable":
        if len(args) < 2:
            print("Usage: schedule enable <id>")
            return
        try:
            task_id = int(args[1])
        except ValueError:
            print(f"Invalid task ID: {args[1]}")
            return
        if schedule_db.set_enabled(task_id, True):
            print(f"Enabled schedule #{task_id}.")
        else:
            print(f"Schedule #{task_id} not found.")

    elif action == "disable":
        if len(args) < 2:
            print("Usage: schedule disable <id>")
            return
        try:
            task_id = int(args[1])
        except ValueError:
            print(f"Invalid task ID: {args[1]}")
            return
        # Cancel running task before disabling
        from src.scheduler.cron_scheduler import get_engine
        get_engine().cancel_task(task_id)
        if schedule_db.set_enabled(task_id, False):
            print(f"Disabled schedule #{task_id}.")
        else:
            print(f"Schedule #{task_id} not found.")

    elif action == "run":
        if len(args) < 2:
            print("Usage: schedule run <id>")
            return
        try:
            task_id = int(args[1])
        except ValueError:
            print(f"Invalid task ID: {args[1]}")
            return
        from src.scheduler.cron_scheduler import get_engine
        from src.device.emulator import emulator_manager
        devices = emulator_manager.list_online or ["emulator-5554"]
        engine = get_engine(device_serials=devices)
        engine.trigger_now(task_id)
        print(f"Triggered schedule #{task_id} to run now.")

    elif action == "daemon":
        import signal
        import os as _os
        from src.device.emulator import emulator_manager
        from src.scheduler.cron_scheduler import get_engine

        # PID file to prevent duplicate daemon processes
        pid_dir = Path(config.DATA_DIR) / "run"
        pid_dir.mkdir(parents=True, exist_ok=True)
        pid_file = pid_dir / "scheduler_daemon.pid"

        if pid_file.exists():
            try:
                old_pid = int(pid_file.read_text().strip())
                # Check if the process is still alive
                try:
                    _os.kill(old_pid, 0)  # Signal 0 = existence check
                    print(f"ERROR: Scheduler daemon is already running (PID {old_pid}).")
                    print(f"  If it's not, delete {pid_file} and try again.")
                    return
                except OSError:
                    # Process not running — stale PID file
                    print(f"Removing stale PID file (PID {old_pid} not running).")
                    pid_file.unlink()
            except (ValueError, FileNotFoundError):
                pass

        pid_file.write_text(str(_os.getpid()))
        print(f"PID file: {pid_file}")

        def _cleanup() -> None:
            """Graceful shutdown: stop engine, stop monitors, remove PID file."""
            print("\nShutting down scheduler...")
            try:
                engine.stop()
            except Exception as e:
                print(f"Engine stop error: {e}")
            try:
                emulator_manager.stop_health_monitor()
            except Exception:
                pass
            try:
                pid_file.unlink(missing_ok=True)
            except Exception:
                pass
            print("Scheduler stopped.")

        def _signal_handler(signum, frame) -> None:
            signame = signal.Signals(signum).name
            print(f"\nReceived {signame} — initiating graceful shutdown...")
            _cleanup()
            _os._exit(0)

        signal.signal(signal.SIGTERM, _signal_handler)
        signal.signal(signal.SIGINT, _signal_handler)

        devices = emulator_manager.list_online
        if not devices:
            print("No ADB devices found. Using default 'emulator-5554'.")
            devices = ["emulator-5554"]

        print(f"Devices: {', '.join(devices)}")
        engine = get_engine(device_serials=devices)
        engine.start()

        # ---- Wire emulator lifecycle → scheduler coordination ----
        def _on_emulator_event(event_type: str, serial: str) -> None:
            if event_type == "pre_restart":
                print(f"\n🔄 模拟器 {serial} 正在重启...")
                engine.pause_device(serial)
            elif event_type == "post_restart":
                print(f"✅ 模拟器 {serial} 重启完成，恢复调度。")
                engine.resume_device(serial)
            elif event_type == "restart_failed":
                print(f"❌ 模拟器 {serial} 重启失败！请手动检查。")
                engine.resume_device(serial)  # Resume anyway so manual tasks can try
            elif event_type == "disconnected":
                print(f"⚠️  模拟器 {serial} 连接断开。")
            elif event_type == "reconnected":
                print(f"🔗 模拟器 {serial} 重新连接。")

        emulator_manager.on_health_event(_on_emulator_event)

        # Start health monitoring (includes memory watchdog + scheduled restart)
        for serial in devices:
            emulator_manager.start_health_monitor(serial)

        # Show initial memory status
        mem = emulator_manager.get_emulator_memory_mb()
        if mem:
            total_mb = sum(mem.values())
            print(f"模拟器内存: {total_mb}MB / {config.emulator.memory_limit_mb}MB 上限")

        print("Scheduler daemon is running. Press Ctrl+C to stop.")

        try:
            while True:
                time.sleep(10)
        except KeyboardInterrupt:
            _cleanup()

    else:
        print(f"Unknown action: {action}")
        print("Available: list, add, remove, enable, disable, run, daemon")


def cmd_emulator(args: list[str]) -> None:
    """Manage emulator lifecycle: status, restart, memory.

    Usage:
        emulator status                     # Show emulator health + memory
        emulator restart [<serial>]         # Restart a specific device (or all)
        emulator memory                     # Show memory breakdown
    """
    from config.settings import config
    from src.device.emulator import emulator_manager

    action = args[0] if args else "status"

    if action == "status":
        status = emulator_manager.health_status
        devices = status.get("devices", {})
        mem = status.get("emulator_memory", {})

        if not devices:
            print("No devices are being monitored.")
            return

        print("=== Emulator Status ===")
        for serial, info in devices.items():
            online = "✅" if info["online"] else "❌"
            restarting = " 🔄 RESTARTING" if info.get("restarting") else ""
            last = info.get("last_restart") or "never"
            print(f"  {serial}: online={online} failures={info['consecutive_failures']}"
                  f" last_restart={last}{restarting}")

        print(f"\n=== Memory ({config.emulator.type} @ {config.emulator.instance_name}) ===")
        processes = mem.get("processes", {})
        if processes:
            for name, mb in processes.items():
                gb = mb / 1024.0
                bar = _memory_bar(mb, config.emulator.memory_limit_mb)
                print(f"  {name}: {mb}MB ({gb:.1f}GB) {bar}")
            print(f"  TOTAL: {mem['total_mb']}MB ({mem['total_gb']}GB) "
                  f"/ limit={mem['limit_mb']}MB")
        else:
            print("  (psutil not installed or no emulator processes found)")
            print("  Install with: pip install psutil")

        print(f"\n=== Lifecycle Config ===")
        restart_cron = config.emulator.restart_cron or "disabled"
        limit_gb = config.emulator.memory_limit_mb / 1024.0
        if config.emulator.memory_limit_mb > 0:
            print(f"  Memory limit: {config.emulator.memory_limit_mb}MB ({limit_gb:.1f}GB)")
        else:
            print(f"  Memory limit: disabled (set EMULATOR_MEMORY_LIMIT_MB to enable)")
        print(f"  Scheduled restart: {restart_cron}")
        print(f"  Console: {config.emulator.console_path}")

    elif action == "restart":
        serial = args[1] if len(args) > 1 else emulator_manager.first_online
        if not serial:
            print("No device available to restart.")
            return

        print(f"Restarting emulator for {serial}...")
        result = emulator_manager.restart_emulator(serial)
        if result == "ok":
            print(f"✅ {serial} restarted successfully.")
        elif result == "already_running":
            print(f"⏳ {serial} restart already in progress.")
        else:
            print(f"❌ {serial} restart failed. Check logs for details.")

    elif action == "memory":
        mem = emulator_manager.get_emulator_memory_mb()
        total = sum(mem.values())
        limit = config.emulator.memory_limit_mb
        if not mem:
            print("No emulator processes detected (psutil may not be installed).")
            return
        print(f"Emulator memory ({config.emulator.type}):")
        for name, mb in mem.items():
            bar = _memory_bar(mb, limit)
            print(f"  {name}: {mb}MB ({mb/1024:.1f}GB) {bar}")
        print(f"  TOTAL: {total}MB ({total/1024:.1f}GB)  limit={limit}MB")

    else:
        print(f"Unknown action: {action}")
        print("Available: status, restart, memory")


def _memory_bar(mb: int, limit: int, width: int = 20) -> str:
    """Draw a simple ASCII memory bar."""
    if limit <= 0:
        return ""
    ratio = min(mb / limit, 1.0)
    filled = int(ratio * width)
    bar = "█" * filled + "░" * (width - filled)
    pct = int(ratio * 100)
    return f"[{bar}] {pct}%"


def main() -> None:
    setup_logging()
    setup_container()
    args = sys.argv[1:]

    if not args:
        print("Usage: python -m src.cli [chat|run|skills|status|weixin|schedule|emulator]")
        return

    subcommand = args[0]
    sub_args = args[1:]

    if subcommand == "chat":
        cmd_chat(sub_args)
    elif subcommand == "run":
        cmd_run(sub_args)
    elif subcommand == "skills":
        cmd_skills(sub_args)
    elif subcommand == "status":
        cmd_status(sub_args)
    elif subcommand == "weixin":
        cmd_weixin(sub_args)
    elif subcommand == "schedule":
        cmd_schedule(sub_args)
    elif subcommand == "emulator":
        cmd_emulator(sub_args)
    else:
        print(f"Unknown command: {subcommand}")
        print("Available: chat, run, skills, status, weixin, schedule, emulator")


if __name__ == "__main__":
    main()
