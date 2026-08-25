#!/usr/bin/env python3
"""spaceThink MVP — Master Application Stack Runner.

Launches the entire spaceThink ecosystem in a single command:
1. Verifies/generates synthetic telemetry and closed-loop diagnostic reports
2. Starts the FastAPI REST API server (http://127.0.0.1:8000)
3. Starts the Streamlit EXHYTE Dashboard (http://localhost:8501)
4. Manages process lifecycle with colorized streaming logs and graceful shutdown.

Usage:
    python run.py
    python run.py --dashboard-only
    python run.py --api-only
    python run.py --port-api 8000 --port-dashboard 8501
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

# Ensure UTF-8 output encoding across Windows / Linux
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Color styling for terminal output
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
BLUE = "\033[94m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"

# Enable VT100 escape codes on Windows console if supported
if sys.platform == "win32":
    try:
        os.system("color")
    except Exception:
        pass


def find_project_root() -> Path:
    """Find the spacethink-mvp directory regardless of where script is called from."""
    current = Path(__file__).resolve().parent
    if (current / "api" / "app.py").exists() and (current / "dashboard" / "app.py").exists():
        return current
    if (current / "spacethink-mvp" / "api" / "app.py").exists():
        return current / "spacethink-mvp"
    return current


def find_python_executable(root_dir: Path) -> str:
    """Find the appropriate python executable (prefer venv if available)."""
    # Check for local venv in root_dir
    windows_venv = root_dir / ".venv" / "Scripts" / "python.exe"
    posix_venv = root_dir / ".venv" / "bin" / "python"

    if windows_venv.exists():
        return str(windows_venv)
    if posix_venv.exists():
        return str(posix_venv)

    # Check parent venv
    parent_win = root_dir.parent / ".venv" / "Scripts" / "python.exe"
    parent_posix = root_dir.parent / ".venv" / "bin" / "python"
    if parent_win.exists():
        return str(parent_win)
    if parent_posix.exists():
        return str(parent_posix)

    return sys.executable


def stream_process_output(pipe, prefix: str, color: str):
    """Stream subprocess stdout/stderr line by line with a styled tag."""
    try:
        for raw_line in iter(pipe.readline, ""):
            if not raw_line:
                break
            line = raw_line.rstrip()
            if line:
                try:
                    print(f"{color}{BOLD}[{prefix}]{RESET} {line}", flush=True)
                except Exception:
                    pass
    except Exception:
        pass
    finally:
        try:
            pipe.close()
        except Exception:
            pass


def ensure_data_prepared(python_bin: str, root_dir: Path, force: bool = False):
    """Check if synthetic data and reports.json exist; generate them if missing."""
    synthetic_dir = root_dir / "data" / "synthetic"
    reports_file = root_dir / "data" / "reports.json"

    has_data = synthetic_dir.exists() and any(synthetic_dir.glob("*.csv"))
    has_reports = reports_file.exists() and reports_file.stat().st_size > 0

    if not has_data or not has_reports or force:
        print(f"\n{CYAN}{BOLD}spaceThink Data Initialization...{RESET}")
        
        if not has_data or force:
            print(f"{YELLOW}* Generating baseline synthetic reaction-wheel telemetry...{RESET}")
            res = subprocess.run(
                [python_bin, "-m", "cli.main", "generate-data"],
                cwd=str(root_dir),
                capture_output=True,
                text=True,
            )
            if res.returncode != 0:
                print(f"{RED}Failed to generate synthetic data: {res.stderr}{RESET}")
            else:
                print(f"{GREEN}* Synthetic telemetry dataset created in data/synthetic/{RESET}")

        if not has_reports or force:
            print(f"{YELLOW}* Running closed-loop EXHYTE diagnosis across dataset...{RESET}")
            res = subprocess.run(
                [python_bin, "-m", "cli.main", "run-all"],
                cwd=str(root_dir),
                capture_output=True,
                text=True,
            )
            if res.returncode != 0:
                print(f"{RED}Failed to generate reports: {res.stderr}{RESET}")
            else:
                print(f"{GREEN}* Diagnostic reports compiled to data/reports.json{RESET}")


def print_banner(host: str, port_api: int, port_dashboard: int, api_enabled: bool, dashboard_enabled: bool):
    """Print a clean dashboard banner with access URLs."""
    divider = "=" * 76
    print(f"\n{CYAN}{BOLD}{divider}")
    print(f"       spaceThink -- Autonomous EXHYTE Diagnostic & Discovery Agent")
    print(f"{divider}{RESET}\n")

    if api_enabled:
        print(f"  {BLUE}{BOLD}[REST API Server]{RESET}        http://{host}:{port_api}")
        print(f"    - Swagger / OpenAPI:    http://{host}:{port_api}/docs")
        print(f"    - ReDoc Documentation:  http://{host}:{port_api}/redoc")
        print(f"    - Health Check:         http://{host}:{port_api}/v1/health")

    if dashboard_enabled:
        print(f"\n  {GREEN}{BOLD}[EXHYTE UI Dashboard]{RESET}    http://localhost:{port_dashboard}")
        print(f"    - Live Telemetry Multi-Channel Explorer")
        print(f"    - LLM Council Deliberation Panel")
        print(f"    - Human-in-the-Loop Validation Gate")
        print(f"    - Groq Closed-Loop Scientific Timeline")

    print(f"\n{YELLOW}{BOLD}Press Ctrl+C to stop all services.{RESET}\n")


def run_stack(
    host: str = "127.0.0.1",
    port_api: int = 8000,
    port_dashboard: int = 8501,
    api_only: bool = False,
    dashboard_only: bool = False,
    open_browser: bool = True,
    force_generate_data: bool = False,
):
    """Run the API server and Streamlit dashboard concurrently with clean process management."""
    root_dir = find_project_root()
    os.chdir(root_dir)
    python_bin = find_python_executable(root_dir)

    api_enabled = not dashboard_only
    dashboard_enabled = not api_only

    # Prepare datasets if needed
    ensure_data_prepared(python_bin, root_dir, force=force_generate_data)

    print_banner(host, port_api, port_dashboard, api_enabled, dashboard_enabled)

    processes: list[subprocess.Popen] = []
    threads: list[threading.Thread] = []

    def terminate_all():
        print(f"\n{YELLOW}{BOLD}Shutting down spaceThink application stack...{RESET}")
        for p in processes:
            if p.poll() is None:
                try:
                    if sys.platform == "win32":
                        p.terminate()
                    else:
                        p.send_signal(signal.SIGINT)
                except Exception:
                    pass
        time.sleep(0.5)
        for p in processes:
            if p.poll() is None:
                try:
                    p.kill()
                except Exception:
                    pass
        print(f"{GREEN}* All services stopped cleanly.{RESET}")

    # Set up signal handling
    def sig_handler(signum, frame):
        terminate_all()
        sys.exit(0)

    try:
        signal.signal(signal.SIGINT, sig_handler)
        signal.signal(signal.SIGTERM, sig_handler)
    except Exception:
        pass

    try:
        # Start FastAPI REST server
        if api_enabled:
            api_cmd = [
                python_bin,
                "-m",
                "uvicorn",
                "api.app:app",
                "--host",
                host,
                "--port",
                str(port_api),
                "--log-level",
                "info",
            ]
            api_proc = subprocess.Popen(
                api_cmd,
                cwd=str(root_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            processes.append(api_proc)

            t_api = threading.Thread(
                target=stream_process_output,
                args=(api_proc.stdout, "API", BLUE),
                daemon=True,
            )
            t_api.start()
            threads.append(t_api)

        # Start Streamlit Dashboard
        if dashboard_enabled:
            # Let API start up briefly first
            if api_enabled:
                time.sleep(1.2)

            dashboard_cmd = [
                python_bin,
                "-m",
                "streamlit",
                "run",
                "dashboard/app.py",
                "--server.port",
                str(port_dashboard),
                "--server.address",
                "localhost",
                "--server.headless",
                "true",
                "--browser.gatherUsageStats",
                "false",
            ]
            dash_proc = subprocess.Popen(
                dashboard_cmd,
                cwd=str(root_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            processes.append(dash_proc)

            t_dash = threading.Thread(
                target=stream_process_output,
                args=(dash_proc.stdout, "DASHBOARD", GREEN),
                daemon=True,
            )
            t_dash.start()
            threads.append(t_dash)

        # Optionally open the browser
        if open_browser and dashboard_enabled:
            def _open():
                time.sleep(2.5)
                try:
                    webbrowser.open(f"http://localhost:{port_dashboard}")
                except Exception:
                    pass
            threading.Thread(target=_open, daemon=True).start()

        # Keep parent alive and monitor child processes
        while True:
            alive_count = 0
            for p in processes:
                exit_code = p.poll()
                if exit_code is None:
                    alive_count += 1
                elif exit_code != 0:
                    print(f"{RED}Process exited with code {exit_code}{RESET}")
            if alive_count == 0:
                break
            time.sleep(0.5)

    except KeyboardInterrupt:
        pass
    finally:
        terminate_all()


def main():
    parser = argparse.ArgumentParser(
        description="spaceThink MVP — Run the entire closed-loop agent and UI stack.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind REST API to")
    parser.add_argument("--port-api", type=int, default=8000, help="Port for FastAPI server")
    parser.add_argument("--port-dashboard", type=int, default=8501, help="Port for Streamlit dashboard")
    parser.add_argument("--api-only", action="store_true", help="Launch only the FastAPI server")
    parser.add_argument("--dashboard-only", action="store_true", help="Launch only the Streamlit dashboard")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open web browser")
    parser.add_argument("--generate-data", action="store_true", help="Force regenerate synthetic datasets and reports")

    args = parser.parse_args()

    run_stack(
        host=args.host,
        port_api=args.port_api,
        port_dashboard=args.port_dashboard,
        api_only=args.api_only,
        dashboard_only=args.dashboard_only,
        open_browser=not args.no_browser,
        force_generate_data=args.generate_data,
    )


if __name__ == "__main__":
    main()
