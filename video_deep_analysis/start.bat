@echo off
setlocal
cd /d "%~dp0"
set "VENV=%LOCALAPPDATA%\VideoDeepAnalysis\venv"
if not exist "%VENV%\Scripts\python.exe" (
  echo Virtual environment is missing. Run setup.ps1 first.
  pause
  exit /b 1
)
start "Video Deep Analysis" http://127.0.0.1:8000
"%VENV%\Scripts\python.exe" -m uvicorn app:app --host 127.0.0.1 --port 8000
pause
