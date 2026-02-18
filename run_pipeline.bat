@echo off
:: 设置字符集为 UTF-8 以支持中文显示
chcp 65001 >nul
setlocal
title Bilibili Summarizer V3 - Auto Run

:: 自动建立日志目录
if not exist "output\logs" mkdir output\logs
echo [%date% %time%] --- Starting New Run --- >> output\logs\auto_run.log

:: 设置项目路径 (默认为当前目录)
set PROJECT_DIR=%~dp0
set VENV_PATH=%PROJECT_DIR%venv

echo ============================================================
echo      Bilibili Summarizer V3 - 一键批量处理
echo ============================================================
echo.

:: 检查路径是否存在
if not exist "%PROJECT_DIR%" (
    echo [错误] 找不到项目目录: %PROJECT_DIR%
    pause
    exit /b 1
)

cd /d "%PROJECT_DIR%"

:: 激活虚拟环境
echo [1/2] 正在激活虚拟环境...
if exist "%VENV_PATH%\Scripts\activate.bat" (
    call "%VENV_PATH%\Scripts\activate.bat"
) else (
    echo [错误] 找不到虚拟环境: %VENV_PATH%
    pause
    exit /b 1
)

:: 运行流水线，同时输出到控制台和日志文件
echo [2/2] 正在启动全自动流水线...
echo ------------------------------------------------------------
python main.py run >> output\logs\auto_run.log 2>&1
echo ------------------------------------------------------------

echo.
echo [%date% %time%] 任务执行完毕！ >> output\logs\auto_run.log
echo 任务执行完毕！
echo 已上传的书籍可以在微信读书中查看。
echo.
:: 定时任务不需要 pause，手动运行时如有需要可自行添加
:: pause
