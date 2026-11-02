# Kessler AI - Windows PowerShell Launcher
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "       KESSLER AI: ORBITAL SURVIVAL GAME" -ForegroundColor Yellow
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "[ERROR] Python was not found in PATH." -ForegroundColor Red
    Write-Host "Please install Python 3.10+ and add it to PATH." -ForegroundColor Red
    pause
    exit 1
}

Write-Host "Starting game..." -ForegroundColor Green
python main.py $args
