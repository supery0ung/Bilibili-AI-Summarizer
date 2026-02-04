@echo off
:: 设置字符集为 UTF-8 以支持中文显示
chcp 65001 >nul
setlocal
title Bilibili Summarizer V3 - Auto Run

:: 自动获取当前脚本所在目录作为项目目录
set PROJECT_DIR=%~dp0
set PROJECT_DIR=%PROJECT_DIR:~0,-1%

:: 默认虚拟环境路径（假设在项目目录下的 venv）
set VENV_PATH=%PROJECT_DIR%\venv

echo ============================================================
echo      Bilibili Summarizer V3 - 一键批量处理 (MAX=10)
echo ============================================================
echo.

cd /d "%PROJECT_DIR%"

:: 激活虚拟环境
echo [1/2] 正在激活虚拟环境...
if exist "%VENV_PATH%\Scripts\activate.bat" (
    call "%VENV_PATH%\Scripts\activate.bat"
) else (
    echo [错误] 找不到虚拟环境: %VENV_PATH%
    echo 请确保已运行 pip install -r requirements.txt
    pause
    exit /b 1
)

:: 运行流水线
echo [2/2] 正在启动全自动流水线 (处理 10 个视频)...
echo ------------------------------------------------------------
python main.py run --max-items 10
echo ------------------------------------------------------------

echo.
echo 任务执行完毕！
pause
