# -*- coding: utf-8 -*-
"""端到端自检：本地加载 Qwen3-VL 模型 -> 对一张测试图生成 caption

用途：在启动 Gradio 之前，先确认模型能加载、推理能跑通，
避免把环境/版本问题带到 Web UI 里排查。
"""
import os
import sys
import time

# 项目根目录（本脚本所在目录）
ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, "models", "Qwen3-VL-8B-Instruct")

print("=" * 60)
print("步骤 1/4  检查模型目录")
print("=" * 60)
if not os.path.exists(os.path.join(MODEL_DIR, "config.json")):
    print("[错误] 未找到本地模型目录:", MODEL_DIR)
    sys.exit(1)
print("模型目录:", MODEL_DIR)

print()
print("=" * 60)
print("步骤 2/4  加载模型（约 1-2 分钟，首次会读 16GB 权重）")
print("=" * 60)
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

t0 = time.time()
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_DIR,
    dtype=torch.float16,          # 半精度，省显存
    device_map="auto",            # 自动分配到 GPU
    attn_implementation="eager",  # Windows 用 eager，无需编译 flash-attn
)
processor = AutoProcessor.from_pretrained(MODEL_DIR)
print("加载完成，用时 %.1f 秒" % (time.time() - t0))
print("模型所在设备:", next(model.parameters()).device)
if torch.cuda.is_available():
    print("显存占用: %.2f GB" % (torch.cuda.memory_allocated() / 1073741824))

print()
print("=" * 60)
print("步骤 3/4  生成一张测试图片")
print("=" * 60)
from PIL import Image, ImageDraw
img = Image.new("RGB", (640, 480), (135, 206, 235))   # 天蓝色背景
d = ImageDraw.Draw(img)
d.ellipse([220, 120, 420, 320], fill=(255, 215, 0))   # 黄色圆（太阳）
d.rectangle([0, 380, 640, 480], fill=(34, 139, 34))   # 绿色地面
d.polygon([(120, 380), (200, 240), (280, 380)], fill=(139, 69, 19))  # 棕色山峰
test_img_path = os.path.join(ROOT, "_test_image.jpg")
img.save(test_img_path)
print("测试图已生成:", test_img_path)

print()
print("=" * 60)
print("步骤 4/4  推理生成 caption")
print("=" * 60)
from qwen_vl_utils import process_vision_info

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": test_img_path},
            {"type": "text", "text": "Describe this image in one detailed sentence."},
        ],
    }
]
# 构造对话模板
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
).to("cuda")

t0 = time.time()
with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=128)
# 去掉输入部分，只保留新生成的 token
trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
caption = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
print("推理用时: %.1f 秒" % (time.time() - t0))
print()
print("生成结果:")
print("  ", caption)

# 清理测试图
try:
    os.remove(test_img_path)
except OSError:
    pass

print()
print("=" * 60)
print("[成功] 端到端流程跑通，环境完全可用")
print("=" * 60)
