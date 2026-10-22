# -*- coding: utf-8 -*-
"""旁白音轨专项测试：不保留原声，只听 TTS，验证每条字幕的语音是否真的在对应时间点。"""
import os
import subprocess

import burner

WORK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_burn_debug")
os.makedirs(WORK, exist_ok=True)

video = os.path.join(WORK, "test.mp4")
srt = os.path.join(WORK, "test.srt")
out = os.path.join(WORK, "test_纯旁白.mp4")

# 三条字幕：0s / 2.5s / 5s，拉长到 9 秒视频更好分辨
with open(srt, "w", encoding="utf-8") as f:
    f.write("1\n00:00:00,000 --> 00:00:02,000\n第一句。\n\n"
            "2\n00:00:02,500 --> 00:00:04,500\n第二句。\n\n"
            "3\n00:00:05,000 --> 00:00:07,000\n第三句。\n")

burner.burn_subtitles_with_tts(
    video, srt, out,
    engine="edge", font_size=20, subtitle_position="底部居中",
    original_volume=0.0, narration_volume=1.0,
    keep_original_audio=False,
)

print("\n=== 静音检测（noise=-40dB, d=0.3） ===")
r = subprocess.run(
    [burner.FFMPEG, "-i", out, "-af", "silencedetect=noise=-40dB:d=0.3", "-f", "null", "-"],
    capture_output=True, text=True, encoding="utf-8", errors="ignore")
for line in r.stderr.splitlines():
    if "silence_" in line:
        print(line.strip())
