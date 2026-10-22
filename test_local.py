# -*- coding: utf-8 -*-
"""本机部署自检脚本：验证 4B + 4bit 能否在 GTX 1660(6GB) 上加载并推理。

与仓库自带的 test_run.py 区别：本脚本强制使用 Qwen3-VL-4B + 4-bit，
且直接复用 app.py 的 load_selected_model()，保证验证的就是界面实际走的代码路径。
"""
import os
import time

from PIL import Image, ImageDraw

import app as cap  # 复用 app.py 的加载与推理函数


def main():
    # 造一张带文字的测试图，避免依赖外部素材
    img = Image.new("RGB", (640, 400), (30, 60, 120))
    d = ImageDraw.Draw(img)
    d.rectangle([80, 120, 560, 280], fill=(250, 250, 250))
    d.text((110, 180), "SIMPLE CAPTIONER TEST", fill=(0, 0, 0))
    test_img = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_selftest.jpg")
    img.save(test_img, quality=95)

    print(f"[1/3] 加载模型 {cap.DEFAULT_MODEL_ID} ({cap.DEFAULT_QUANT}) ...")
    t0 = time.time()
    name, device, vram, dtype, cfg = cap.load_selected_model(
        cap.DEFAULT_MODEL_ID, cap.DEFAULT_QUANT, cap.DEFAULT_ATTN
    )
    print(f"      加载耗时 {time.time() - t0:.1f}s")
    print(f"      设备={device} 显存占用={vram} dtype={dtype}")

    print("[2/3] 推理一张图 ...")
    t1 = time.time()
    out = cap.generate_caption(
        test_img,
        prompt="Describe this image in one sentence.",
        max_tokens=64,
        resolution_mode="auto",
        subtitle_mode=False,
    )
    print(f"      推理耗时 {time.time() - t1:.1f}s")
    print(f"      输出: {out}")

    print("[3/3] 结果校验 ...")
    ok = isinstance(out, str) and len(out.strip()) > 3
    print("      PASS" if ok else "      FAIL：模型未返回有效文本")

    # 清理测试图
    try:
        os.remove(test_img)
    except OSError:
        pass


if __name__ == "__main__":
    main()
