# -*- coding: utf-8 -*-
"""从 ModelScope（魔搭）下载 Qwen-VL 模型到本地 models/ 目录。

背景：本机网络无法访问 HuggingFace，因此改用 ModelScope 拉取官方模型权重。
下载完成后 app.py 会自动优先加载 models/ 下的同名字目录。

用法：
    python download_model.py                              # 默认 Qwen3-VL-8B-Instruct
    python download_model.py Qwen/Qwen3-VL-4B-Instruct    # 指定其它模型
"""
import os
import sys

from modelscope import snapshot_download

# 默认模型；可通过命令行第一个参数覆盖
DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"


def main():
    # 取命令行参数，无参数时用默认模型
    model_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_ID

    # 本地目录名取 ID 最后一段（Qwen/Qwen3-VL-8B-Instruct -> Qwen3-VL-8B-Instruct）
    # 该命名必须与 app.py 中 resolve_model_path() 的约定保持一致
    local_name = model_id.split("/")[-1]
    target_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "models", local_name
    )

    print(f"[下载] 模型 ID : {model_id}")
    print(f"[目标] 保存到 : {target_dir}")
    print("[提示] 首次下载约 17GB，ModelScope 支持断点续传，中断后重跑即可续传")

    # snapshot_download 会拉取仓库全部文件（含 safetensors 权重），并自动处理分片
    path = snapshot_download(model_id, local_dir=target_dir)
    print(f"[完成] 模型已就绪：{path}")


if __name__ == "__main__":
    main()
