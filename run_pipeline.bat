@echo off
chcp 65001 >nul
setlocal
title Bilibili Summarizer V3 - Auto Run

set "PROJECT_DIR=%~dp0"
set "VENV_PATH=%PROJECT_DIR%venv"
set "LOG_DIR=%PROJECT_DIR%output\logs"
set "LOG_FILE=%LOG_DIR%\auto_run.log"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
echo [%date% %time%] --- Starting New Run --- >> "%LOG_FILE%"

cd /d "%PROJECT_DIR%"

if exist "%VENV_PATH%\Scripts\activate.bat" (
    call "%VENV_PATH%\Scripts\activate.bat"
) else (
    echo [ERROR] Virtual environment not found: %VENV_PATH%
    echo Create one with: python -m venv venv
    pause
    exit /b 1
)

echo Running full pipeline...
python main.py run >> "%LOG_FILE%" 2>&1
set "EXIT_CODE=%ERRORLEVEL%"

echo [%date% %time%] Pipeline exited with code %EXIT_CODE% >> "%LOG_FILE%"
exit /b %EXIT_CODE%
