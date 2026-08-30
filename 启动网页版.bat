@echo off
setlocal
cd /d "%~dp0"
set "PROJECT_PYTHON=%~dp0.venv\Scripts\python.exe"
if exist "%PROJECT_PYTHON%" (
  "%PROJECT_PYTHON%" "%~dp0start.py" %*
) else (
  py -3.11 "%~dp0start.py" %*
)
if errorlevel 1 pause
