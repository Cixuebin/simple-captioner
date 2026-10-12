@echo off
chcp 65001 >nul
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo [错误] 未找到虚拟环境 venv，请先双击运行 setup.bat
    pause
    exit /b 1
)

echo ============================================
echo   Simple Captioner 启动中 ...
echo   加载模型需要一些时间，请等待浏览器自动打开
echo ============================================
venv\Scripts\python.exe app.py

pause
