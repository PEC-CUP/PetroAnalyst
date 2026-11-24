from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path


def _bundle_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS")).resolve()
    return Path(__file__).resolve().parent


def _writable_env_file() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        directory = Path(local_app_data) / "PetroAnalyst"
    else:
        directory = Path.home() / ".petroanalyst"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / ".env"


def _available_port(preferred: int) -> int:
    for port in range(preferred, preferred + 40):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("No free local port was found.")


def _open_when_ready(url: str) -> None:
    health_url = url.rstrip("/") + "/health"
    for _ in range(60):
        try:
            with urllib.request.urlopen(health_url, timeout=1) as response:
                if response.status == 200:
                    webbrowser.open(url)
                    return
        except Exception:
            time.sleep(0.5)


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the local PetroAnalyst workspace.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    root = _bundle_root()
    os.chdir(root)
    os.environ.setdefault("PETROANALYST_ENV_FILE", str(_writable_env_file()))

    import uvicorn
    from app.main import app

    port = _available_port(args.port)
    url = f"http://127.0.0.1:{port}/"
    print("=" * 68)
    print("PetroAnalyst is running locally.")
    print(url)
    print("Configure the reviewer API key in the web interface.")
    print("Close this window to stop the service.")
    print("=" * 68)
    if not args.no_browser:
        threading.Thread(target=_open_when_ready, args=(url,), daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
