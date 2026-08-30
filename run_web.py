"""Run the browser-based video analysis service on Windows or macOS."""
from __future__ import annotations

import threading
import webbrowser

import uvicorn


def open_browser() -> None:
    webbrowser.open_new("http://127.0.0.1:8000")


if __name__ == "__main__":
    threading.Timer(0.8, open_browser).start()
    uvicorn.run("app:app", host="127.0.0.1", port=8000)
