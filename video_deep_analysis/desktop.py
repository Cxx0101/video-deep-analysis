"""Windows desktop entry point: local FastAPI service in a native web window."""
from __future__ import annotations

import socket
import sys
import threading
import time
import traceback
from pathlib import Path
from urllib.request import urlopen

import uvicorn
import webview


class DesktopApi:
    def download(self, relative_url: str) -> str:
        """Save a generated MP4 using the native desktop download flow."""
        filename = "视频深度分析结果.mp4"
        target_dir = Path.home() / "Downloads"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / filename
        number = 2
        while target.exists():
            target = target_dir / f"视频深度分析结果 ({number}).mp4"
            number += 1
        with urlopen(f"http://127.0.0.1:8001{relative_url}", timeout=120) as response:
            target.write_bytes(response.read())
        return str(target)


def main() -> None:
    from app import app

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=8001, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        try:
            with socket.create_connection(("127.0.0.1", 8001), timeout=0.2):
                break
        except OSError:
            time.sleep(0.1)
    else:
        raise RuntimeError("本地服务启动超时")
    try:
        webview.create_window("视频深度分析工具", "http://127.0.0.1:8001", js_api=DesktopApi(), width=1000, height=760, min_size=(760, 560))
        webview.start()
    finally:
        server.should_exit = True


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log_dir = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
        (log_dir / "desktop-error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise
