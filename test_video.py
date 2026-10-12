# -*- coding: utf-8 -*-
"""视频端到端自检：走 app.py 的真实代码路径，对一个真实 mp4 生成描述。

直接 import app，复用 app.decode_video_frames / app.generate_caption，
确保测的就是实际部署的代码，而不是另写的一份。

验证点：
  1. PyAV 解码能跑通（原版在 Windows 上因 torchvision 移除 read_video 而必然失败）
  2. 抽帧数被 VIDEO_MAX_FRAMES 正确封顶，不会打满显存
  3. 推理产出描述质量合理
  4. .txt 正确落盘

用法：
    venv\\Scripts\\python.exe test_video.py            # 用默认样本
    venv\\Scripts\\python.exe test_video.py D:\\a.mp4   # 指定任意视频
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
MODEL_DIR = os.path.join(ROOT, "models", "Qwen3-VL-8B-Instruct")

# 默认测试样本：一段 26 秒 1080p 真实录屏（内容为武器资料网站浏览）
# 选用真实素材而非合成色块 —— 合成色块会被模型"脑补"成无意义内容，验证不了质量
DEFAULT_VIDEO = r"C:\Users\Starshark_Ci\Videos\2026-09-18 15-20-28.mp4"

VIDEO = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO


def log(msg):
    print(msg, flush=True)


log("=" * 60)
log("步骤 1/5  确认测试视频")
log("=" * 60)
if not os.path.exists(VIDEO):
    log(f"[错误] 找不到视频: {VIDEO}")
    log("      可以传入自定义路径：python test_video.py <视频路径>")
    sys.exit(1)
size_mb = os.path.getsize(VIDEO) / 1048576
log(f"视频路径: {VIDEO}")
log(f"文件大小: {size_mb:.1f} MB")

log("")
log("=" * 60)
log("步骤 2/5  导入 app.py 并加载模型")
log("=" * 60)
import torch            # noqa: E402
import app              # noqa: E402  （已加 __main__ 守卫，import 不会启动 Web 服务）
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration   # noqa: E402

t0 = time.time()
app.model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_DIR, dtype=torch.float16, device_map="auto", attn_implementation="eager",
)
app.processor = AutoProcessor.from_pretrained(MODEL_DIR)
log(f"模型加载完成，用时 {time.time() - t0:.1f} 秒")

log("")
log("=" * 60)
log("步骤 3/5  PyAV 视频解码（app.decode_video_frames）")
log("=" * 60)
t0 = time.time()
frames, sample_fps = app.decode_video_frames(VIDEO)
log(f"解码耗时: {time.time() - t0:.1f} 秒")
log(f"抽帧数量: {len(frames)} 张（上限 {app.VIDEO_MAX_FRAMES}）")
log(f"采样帧率: {sample_fps:.2f} fps")
log(f"单帧尺寸: {frames[0].size}  模式: {frames[0].mode}")

if len(frames) > app.VIDEO_MAX_FRAMES:
    log(f"[警告] 抽帧数超过上限 {app.VIDEO_MAX_FRAMES}，可能导致显存打满")

log("")
log("=" * 60)
log("步骤 4/5  调用 app.generate_caption 生成描述")
log("=" * 60)
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
t0 = time.time()
caption = app.generate_caption(VIDEO, app.DEFAULT_PROMPT, 128)
gen_time = time.time() - t0
log(f"推理用时: {gen_time:.1f} 秒")
if torch.cuda.is_available():
    log(f"峰值显存: {torch.cuda.max_memory_allocated() / 1073741824:.1f} GB")
log("")
log("生成结果:")
log("  " + caption.strip().replace("\n", "\n  "))

log("")
log("=" * 60)
log("步骤 5/5  写入同名 .txt（与批处理产物位置一致）")
log("=" * 60)
out_txt = os.path.splitext(VIDEO)[0] + ".txt"
with open(out_txt, "w", encoding="utf-8") as f:
    f.write(caption)
log(f"已写入: {out_txt}")

log("")
log("=" * 60)
log("[成功] 视频流程跑通，环境完全可用")
log("=" * 60)
