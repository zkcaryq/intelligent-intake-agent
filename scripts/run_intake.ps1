param([int]$Port = 8000)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Virtual environment not found. Run: uv venv .venv; uv pip install --python .venv\Scripts\python.exe -r requirements.txt"
}

Set-Location $projectRoot
& $python -m uvicorn main:app --host 127.0.0.1 --port $Port
