@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================
echo   安装依赖（PyTorch CUDA 13.0 + 项目依赖）
echo ============================================

if not exist "venv\Scripts\python.exe" (
    echo [1/3] 创建虚拟环境 ...
    python -m venv venv
) else (
    echo [1/3] 虚拟环境已存在，跳过创建
)

set VP=venv\Scripts\python.exe

echo [2/3] 安装 PyTorch (CUDA 13.0) ...
echo       注意：必须从 download.pytorch.org 装，清华源的 torch 是 CPU-only 版
%VP% -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
if errorlevel 1 goto :fail

echo [3/3] 安装其余依赖（含 modelscope 下载器 与 PyAV 视频解码库）...
%VP% -m pip install "gradio>=6.10.0" "Pillow>=11.2.1" "qwen_vl_utils>=0.0.11" "transformers>=5.4.0" "accelerate>=0.26.0" "bitsandbytes>=0.46.1" "hf_xet>=1.1.10" "av>=12.0.0" modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple
if errorlevel 1 goto :fail

echo.
echo 依赖安装完成！下一步：双击 download_model.bat 下载模型
pause
exit /b 0

:fail
echo.
echo [错误] 依赖安装失败，请检查网络后重试
pause
exit /b 1
