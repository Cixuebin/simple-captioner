# -*- coding: utf-8 -*-
"""烧录失败排查脚本：用合成视频+SRT 直接调用 burner，把真实异常打出来。

用法：
    venv\\Scripts\\python.exe debug_burn.py
"""
import os
import subprocess
import traceback

import burner

WORK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_burn_debug")
os.makedirs(WORK, exist_ok=True)

video = os.path.join(WORK, "test.mp4")
srt = os.path.join(WORK, "test.srt")
out = os.path.join(WORK, "test_旁白版.mp4")

# 1) 用 ffmpeg 造一个 6 秒、带音轨的测试视频
if not os.path.exists(video):
    print("[1] 生成测试视频 ...")
    r = subprocess.run([
        burner.FFMPEG, "-y",
        "-f", "lavfi", "-i", "testsrc=size=640x360:rate=25:duration=6",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=6",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
        video,
    ], capture_output=True, text=True, encoding="utf-8", errors="ignore")
    print("    returncode:", r.returncode)
    if r.returncode != 0:
        print(r.stderr[-1500:])

# 2) 写 SRT
with open(srt, "w", encoding="utf-8") as f:
    f.write("1\n00:00:00,000 --> 00:00:03,000\n这是第一句测试字幕。\n\n"
            "2\n00:00:03,000 --> 00:00:06,000\n这是第二句测试字幕。\n")

print(f"[2] ffmpeg 版本: {burner.ffmpeg_version()}  存在视频: {os.path.exists(video)}")
print(f"    视频有音轨: {burner.has_audio_stream(video)}  时长: {burner.probe_duration(video)}")

# 3) 直接调烧录，完整打印异常
print("[3] 开始烧录 ...")
try:
    burner.burn_subtitles_with_tts(
        video, srt, out,
        engine="edge",
        font_name="Microsoft YaHei",
        font_size=20,
        subtitle_position="底部居中",
        original_volume=0.5,
        narration_volume=1.0,
        keep_original_audio=True,
        outline_enabled=True,
        outline_width=2,
        text_opacity=1.0,
    )
    print("    ✅ 成功:", out, os.path.getsize(out), "bytes")
except Exception:
    print("    ❌ 失败，完整堆栈：")
    traceback.print_exc()
