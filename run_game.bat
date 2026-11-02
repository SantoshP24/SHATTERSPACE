@echo off
title Kessler AI - Orbital Survival Game
echo ========================================================
echo        KESSLER AI: ORBITAL SURVIVAL GAME
echo ========================================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not found in your PATH.
    echo Please install Python 3.10+ and add it to your system PATH.
    pause
    exit /b 1
)

echo Launching Kessler AI...
python main.py %*

if errorlevel 1 (
    echo.
    echo [NOTE] Game terminated with exit code %ERRORLEVEL%.
    pause
)
