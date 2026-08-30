"""One-command bootstrapper for the local web application."""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
VENV_DIR = PROJECT_DIR / ".venv"
REQUIREMENTS = PROJECT_DIR / "requirements.txt"
STAMP = VENV_DIR / ".requirements.sha256"


def venv_python() -> Path:
    return VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def run(command: list[str]) -> None:
    completed = subprocess.run(command, cwd=PROJECT_DIR)
    if completed.returncode:
        raise SystemExit(completed.returncode)


def main() -> None:
    python = venv_python()
    if "--check" in sys.argv:
        print(f"启动环境检查通过：{python}")
        return
    if not python.is_file():
        print("首次启动：正在创建虚拟环境…")
        run([sys.executable, "-m", "venv", str(VENV_DIR)])
        python = venv_python()

    fingerprint = hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest()
    if not STAMP.is_file() or STAMP.read_text(encoding="utf-8") != fingerprint:
        print("首次启动或依赖已更新：正在安装依赖…")
        run([str(python), "-m", "pip", "install", "--upgrade", "pip"])
        run([str(python), "-m", "pip", "install", "-r", str(REQUIREMENTS)])
        STAMP.write_text(fingerprint, encoding="utf-8")

    run([str(python), "run_web.py"])


if __name__ == "__main__":
    main()
