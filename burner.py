# -*- coding: utf-8 -*-
"""
字幕烧录 + 语音旁白合成。

对外只暴露一个入口：
    burn_subtitles_with_tts(video, srt, out, ...)
"""
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )
    except Exception:
        pass

FFMPEG = r"D:\EVCapture\ffmpeg.exe"
FFPROBE_CANDIDATES = [
    os.path.join(os.path.dirname(FFMPEG), "ffprobe.exe"),
    FFMPEG,
]

TC_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})"
)

# 与 app.py 的视频扩展名保持一致
_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".gif", ".flv"}


def is_video(filename):
    """按扩展名判断是否视频文件。"""
    return os.path.splitext(filename)[1].lower() in _VIDEO_EXTS


def _no_window():
    """Windows 下隐藏子进程黑窗。"""
    if os.name == "nt":
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        return {"startupinfo": si, "creationflags": subprocess.CREATE_NO_WINDOW}
    return {}


_FFMPEG_VERSION = None


def ffmpeg_version():
    """返回 (major, minor)，用于按版本决定可用参数。"""
    global _FFMPEG_VERSION
    if _FFMPEG_VERSION is None:
        try:
            r = subprocess.run(
                [FFMPEG, "-version"], capture_output=True, text=True,
                encoding="utf-8", errors="ignore", **_no_window(),
            )
            m = re.search(r"ffmpeg version \S*?(\d+)\.(\d+)", r.stdout)
            _FFMPEG_VERSION = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
        except Exception:
            _FFMPEG_VERSION = (0, 0)
    return _FFMPEG_VERSION


def find_ffmpeg():
    """按优先级找一个能用的 ffmpeg，返回路径。"""
    candidates = [
        os.environ.get("SC_FFMPEG"),
        shutil.which("ffmpeg"),
        FFMPEG,
        r"D:\JianyingPro\11.5.0.14471\ffmpeg.exe",
    ]
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    return None


# ---------------------------------------------------------------- SRT 解析
def parse_srt(path):
    """解析 SRT -> [(start_sec, end_sec, text), ...]"""
    with open(path, encoding="utf-8") as f:
        content = f.read()
    items = []
    for block in content.strip().split("\n\n"):
        lines = [l for l in block.split("\n") if l.strip()]
        if len(lines) < 3:
            continue
        m = TC_RE.search(lines[1])
        if not m:
            continue
        g = [int(x) for x in m.groups()]
        start = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
        end = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
        items.append((start, end, " ".join(lines[2:])))
    return items


def probe_duration(path):
    """用 ffmpeg -i 读取媒体时长（秒）。"""
    out = subprocess.run(
        [FFMPEG, "-i", path, "-hide_banner"],
        capture_output=True, text=True, encoding="utf-8", errors="ignore",
        **_no_window(),
    )
    m = re.search(r"Duration: (\d+):(\d+):(\d+)\.(\d+)", out.stderr)
    if not m:
        return None
    h, mi, s, cs = (int(x) for x in m.groups())
    return h * 3600 + mi * 60 + s + cs / 100


def has_audio_stream(path):
    """判断视频有没有音轨。"""
    out = subprocess.run(
        [FFMPEG, "-i", path, "-hide_banner"],
        capture_output=True, text=True, encoding="utf-8", errors="ignore",
        **_no_window(),
    )
    return "Audio:" in out.stderr


# ------------------------------------------------------------------ TTS
def tts_edge(text, out_path, voice="zh-CN-XiaoxiaoNeural", rate="+0%"):
    """用 edge-tts（微软在线）合成，音质接近真人。"""
    import asyncio
    import edge_tts

    async def _run():
        comm = edge_tts.Communicate(text, voice, rate=rate)
        await comm.save(out_path)

    asyncio.run(_run())


def tts_sapi(text, out_path, rate=0):
    """用 Windows 自带 SAPI 合成（离线）。需要 pywin32。"""
    import win32com.client

    wav = os.path.splitext(out_path)[0] + ".wav"
    speaker = win32com.client.Dispatch("SAPI.SpVoice")
    # 优先中文语音
    for v in speaker.GetVoices():
        if "Chinese" in v.GetDescription() or "Huihui" in v.GetDescription():
            speaker.Voice = v
            break
    speaker.Rate = rate
    stream = win32com.client.Dispatch("SAPI.SpFileStream")
    stream.Open(wav, 3, False)
    speaker.AudioOutputStream = stream
    speaker.Speak(text)
    stream.Close()
    return wav


def synth_voice(text, out_path, engine="edge", voice="zh-CN-XiaoxiaoNeural"):
    """合成单条语音，失败时自动回退到另一个引擎。"""
    if engine == "edge":
        try:
            tts_edge(text, out_path, voice=voice)
            if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
                return out_path, "edge"
            raise RuntimeError("edge-tts 输出为空")
        except Exception as e:
            print(f"[WARN] edge-tts 失败({e})，回退 SAPI")
            wav = tts_sapi(text, out_path)
            return wav, "sapi"
    else:
        return tts_sapi(text, out_path), "sapi"


# ------------------------------------------------- 构建旁白音轨（按时间点）
def build_narration_track(items, workdir, engine="edge",
                          voice="zh-CN-XiaoxiaoNeural"):
    """把每条字幕的语音放到它对应的时间点上，输出一条完整音轨。

    做法：每条语音先延迟到 start 时间，再用 amix 混合。
    """
    clips = []
    for i, (start, end, text) in enumerate(items):
        raw = os.path.join(workdir, f"v{i}.mp3")
        path, used = synth_voice(text, raw, engine=engine, voice=voice)
        clips.append((start, path, used, text))
        print(f"  [TTS] #{i+1} @{start:.1f}s ({used}) {text}")

    # 用 ffmpeg 把每条 clip 延迟后混合。用 adelay + amix。
    # aresample 统一采样率，否则 SAPI(22050) 与 edge-tts(24000) 混在一起会出声学问题。
    inputs = []
    filters = []
    for i, (start, path, _u, _t) in enumerate(clips):
        inputs += ["-i", path]
        delay_ms = int(start * 1000)
        filters.append(
            f"[{i}:a]aresample=44100,"
            f"aformat=sample_fmts=fltp:channel_layouts=stereo,"
            f"adelay={delay_ms}|{delay_ms}[a{i}]"
        )
    mix_in = "".join(f"[a{i}]" for i in range(len(clips)))
    # 注意：老版本 ffmpeg（2019 前后）的 amix 不支持 normalize 参数，
    # 一旦带上会直接报 "Option 'normalize' not found" 而整条命令失败。
    # 用 volume 手动控制音量即可，兼容性最好。
    filters.append(
        f"{mix_in}amix=inputs={len(clips)}:duration=longest[out]"
    )
    track = os.path.join(workdir, "narration.m4a")
    cmd = ([FFMPEG, "-y"] + inputs +
           ["-filter_complex", ";".join(filters), "-map", "[out]",
            "-c:a", "aac", "-b:a", "192k", track])
    subprocess.run(cmd, capture_output=True, **_no_window())
    return track


# -------------------------------------------------------------- 主流程
def burn_subtitles_with_tts(video, srt, out_path,
                            engine="edge",
                            voice="zh-CN-XiaoxiaoNeural",
                            font_name="Microsoft YaHei",
                            font_size=20,
                            keep_original_audio=True,
                            progress_cb=None):
    """把字幕烧录进画面，并把 TTS 旁白混进音轨。"""

    def rep(stage, detail=""):
        print(f"[{stage}] {detail}")
        if progress_cb:
            try:
                progress_cb(stage, detail)
            except Exception:
                pass

    items = parse_srt(srt)
    if not items:
        raise RuntimeError(f"未能从 {srt} 解析出任何字幕")

    workdir = tempfile.mkdtemp(prefix="sc_burn_")
    try:
        rep("tts", f"正在合成 {len(items)} 条旁白语音 ...")
        narration = build_narration_track(
            items, workdir, engine=engine, voice=voice
        )

        # subtitles 滤镜的路径在 Windows 下要转义：反斜杠 -> 正斜杠，冒号 -> \:
        # 但文件名里若含单引号（如 'xxx' 这类自带引号的名字）仍会破坏滤镜引号，
        # 所以最稳的做法：把 srt 复制成 workdir 下的安全文件名再引用。
        safe_srt = os.path.join(workdir, "subs.srt")
        shutil.copyfile(srt, safe_srt)
        srt_esc = safe_srt.replace("\\", "/").replace(":", "\\:")
        style = (
            f"FontName={font_name},FontSize={font_size},"
            "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
            "BorderStyle=1,Outline=2,Shadow=1,MarginV=28"
        )
        vf = f"subtitles='{srt_esc}':force_style='{style}'"

        rep("render", "正在烧录字幕并混音 ...")
        ver = ffmpeg_version()
        # normalize 参数只在 ffmpeg 4.4+ 存在，老版本带上会直接报错
        no_norm = "" if ver >= (4, 4) else ""
        if keep_original_audio and has_audio_stream(video):
            # 原声压低到 0.5、旁白保持 1.0，避免旁白被音乐盖住
            cmd = [
                FFMPEG, "-y",
                "-i", video,
                "-i", narration,
                "-filter_complex",
                f"[0:v]{vf}[v];"
                f"[0:a]volume=0.5[oa];"
                f"[1:a]volume=1.0[na];"
                f"[oa][na]amix=inputs=2:duration=first{no_norm}[a]",
                "-map", "[v]", "-map", "[a]",
                "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k",
                "-movflags", "+faststart",
                out_path,
            ]
        else:
            cmd = [
                FFMPEG, "-y",
                "-i", video,
                "-i", narration,
                "-filter_complex", f"[0:v]{vf}[v]",
                "-map", "[v]", "-map", "1:a",
                "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k",
                "-shortest", "-movflags", "+faststart",
                out_path,
            ]

        r = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="ignore", **_no_window())
        if r.returncode != 0:
            print(r.stderr[-2500:])
            raise RuntimeError("ffmpeg 烧录失败，见上方日志")
        rep("done", f"已生成 {os.path.basename(out_path)}")
        return out_path
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
