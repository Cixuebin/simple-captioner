@echo off
chcp 65001 >nul
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo [错误] 请先运行 setup.bat 安装依赖
    pause
    exit /b 1
)

echo ============================================
echo   从 ModelScope 下载 Qwen3-VL-8B-Instruct
echo   约 17GB，支持断点续传，可关掉重来
echo ============================================
venv\Scripts\python.exe download_model.py Qwen/Qwen3-VL-8B-Instruct

pause
