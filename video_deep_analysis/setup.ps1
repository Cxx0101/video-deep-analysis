$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectDir "runtime\python311\python.exe"
if (-not (Test-Path -LiteralPath $Python)) { throw "未找到项目配套 Python 3.11：$Python" }
Set-Location $ProjectDir
$Venv = Join-Path $env:LOCALAPPDATA "VideoDeepAnalysis\venv"
& $Python -m venv $Venv
& (Join-Path $Venv "Scripts\python.exe") -m pip install --upgrade pip
& (Join-Path $Venv "Scripts\python.exe") -m pip install -r requirements.txt
Write-Host "安装完成。双击 start.bat 启动。"
