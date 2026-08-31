"""Run the browser-based video analysis service on Windows or macOS."""
from __future__ import annotations

import socket
import threading
import webbrowser

import uvicorn


def find_available_port() -> int:
    """Prefer 8000, but avoid accidentally opening an older running copy."""
    for port in range(8000, 8011):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if probe.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("8000 到 8010 端口均被占用，请关闭旧的网页服务后重试。")


def open_browser(port: int) -> None:
    webbrowser.open_new(f"http://127.0.0.1:{port}")


if __name__ == "__main__":
    port = find_available_port()
    threading.Timer(0.8, open_browser, args=(port,)).start()
    uvicorn.run("app:app", host="127.0.0.1", port=port)
