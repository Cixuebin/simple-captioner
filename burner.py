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


# 字幕位置名 → 对齐表（值按 ffmpeg 版本在 _position_params 里选择）
# 新版 ASS（numpad 对齐）：1~3 底部（左/中/右），4~6 中部，7~9 顶部
_MODERN_ALIGN = {
    "底部居中": (2, 28),
    "底部靠左": (1, 28),
    "底部靠右": (3, 28),
    "画面正中": (5, 0),
    "顶部居中": (8, 28),
}
# 旧版 SSA 对齐表（2019 ffmpeg 实测验证）：5~7 顶部，9~11 中部
_LEGACY_ALIGN = {
    "底部居中": (2, 28),
    "底部靠左": (1, 28),
    "底部靠右": (3, 28),
    "画面正中": (10, 0),
    "顶部居中": (6, 28),
}

# 供 Web UI 下拉框展示用（只取键名）
SUBTITLE_POSITIONS = _MODERN_ALIGN


def _position_params(name):
    """把位置名转成 (Alignment, MarginV)，未知值退回底部居中。

    注意：旧版 ffmpeg（< 4.4，如 2019 构建）内部按 SSA 对齐表解释
    Alignment（实测 5=顶部、8=中部），与新版 ASS 不同，
    所以这里按 ffmpeg 版本返回不同的 Alignment 值。
    """
    legacy = ffmpeg_version() < (4, 4)
    table = _LEGACY_ALIGN if legacy else _MODERN_ALIGN
    return table.get(name, table["底部居中"])


def _no_window():
    """Windows 下隐藏子进程黑窗。"""
    if os.name == "nt":
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        return {"startupinfo": si, "creationflags": subprocess.CREATE_NO_WINDOW}
    return {}


_FFMPEG_VERSION = None


def ffmpeg_version():
    """返回 (major, minor)，用于按版本决定可用参数。

    兼容两种版本串：
      - 正式版：  "ffmpeg version 6.1.1 ..."          -> (6, 1)
      - git 快照："ffmpeg version git-2019-10-22-..." -> 按日期年份估算
        （ffmpeg 4.4 于 2021 年发布，git-2020 及更早按旧版处理）
    """
    global _FFMPEG_VERSION
    if _FFMPEG_VERSION is None:
        try:
            r = subprocess.run(
                [FFMPEG, "-version"], capture_output=True, text=True,
                encoding="utf-8", errors="ignore", **_no_window(),
            )
            out = r.stdout
            m = re.search(r"ffmpeg version (\d+)\.(\d+)", out)
            if m:
                _FFMPEG_VERSION = (int(m.group(1)), int(m.group(2)))
            else:
                # git 快照：取 "git-YYYY" 的年份粗略判断新旧
                g = re.search(r"git-(\d{4})", out)
                _FFMPEG_VERSION = ((99, 0) if g and int(g.group(1)) >= 2021
                                   else (0, 0))
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
                            subtitle_position="底部居中",
                            original_volume=0.5,
                            narration_volume=1.0,
                            keep_original_audio=True,
                            outline_enabled=True,
                            outline_width=2,
                            text_opacity=1.0,
                            progress_cb=None):
    """把字幕烧录进画面，并把 TTS 旁白混进音轨。

    font_size         : 字幕字号（ASS 单位，视频高度 1/… 左右，常用 14~48）
    original_volume   : 视频原声音量倍数（0~2，1.0 = 原始大小）
    narration_volume  : TTS 解说音量倍数（0~3，1.0 = 原始大小）
    subtitle_position : 字幕位置，见 SUBTITLE_POSITIONS
    outline_enabled   : 是否给字幕加描边（黑边）
    outline_width     : 描边粗细（0~4，仅 outline_enabled=True 时生效）
    text_opacity      : 字幕不透明度（0~1，0=全透明、1=完全不透明）
    """

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

    # 参数兜底与范围钳制，避免 UI 传进非法值导致 ffmpeg 报错
    try:
        font_size = int(max(8, min(96, int(font_size))))
    except (TypeError, ValueError):
        font_size = 20
    try:
        original_volume = max(0.0, min(3.0, float(original_volume)))
    except (TypeError, ValueError):
        original_volume = 0.5
    try:
        narration_volume = max(0.0, min(3.0, float(narration_volume)))
    except (TypeError, ValueError):
        narration_volume = 1.0
    try:
        outline_width = int(max(0, min(4, int(outline_width))))
    except (TypeError, ValueError):
        outline_width = 2
    try:
        text_opacity = max(0.0, min(1.0, float(text_opacity)))
    except (TypeError, ValueError):
        text_opacity = 1.0

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
        align, margin_v = _position_params(subtitle_position)
        # ASS 颜色格式 &HAABBGGRR，AA 是不透明度且 00=不透明、FF=全透明，需要反过来。
        alpha = int(round((1.0 - text_opacity) * 255))
        alpha = max(0, min(255, alpha))
        # 描边：关掉描边时把 Outline 与 Shadow 都设为 0，避免残留黑边。
        if outline_enabled and outline_width > 0:
            outline_w, shadow_w = outline_width, 0
        else:
            outline_w, shadow_w = 0, 0
        style = (
            f"FontName={font_name},FontSize={font_size},"
            f"PrimaryColour=&H{alpha:02X}FFFFFF,"
            f"OutlineColour=&H{alpha:02X}000000,"
            "BorderStyle=1,"
            f"Outline={outline_w},Shadow={shadow_w},"
            f"Alignment={align},MarginV={margin_v}"
        )
        vf = f"subtitles='{srt_esc}':force_style='{style}'"

        rep("render", "正在烧录字幕并混音 ...")
        ver = ffmpeg_version()
        # normalize 参数只在 ffmpeg 4.4+ 存在，老版本带上会直接报错
        no_norm = "" if ver >= (4, 4) else ""
        if keep_original_audio and has_audio_stream(video):
            # 原声与旁白各自按用户设定的音量混合
            cmd = [
                FFMPEG, "-y",
                "-i", video,
                "-i", narration,
                "-filter_complex",
                f"[0:v]{vf}[v];"
                f"[0:a]volume={original_volume}[oa];"
                f"[1:a]volume={narration_volume}[na];"
                f"[oa][na]amix=inputs=2:duration=first{no_norm}[a]",
                "-map", "[v]", "-map", "[a]",
                "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k",
                "-movflags", "+faststart",
                out_path,
            ]
        else:
            # 不保留原声（或视频本就没有音轨）时，只用旁白
            cmd = [
                FFMPEG, "-y",
                "-i", video,
                "-i", narration,
                "-filter_complex",
                f"[0:v]{vf}[v];[1:a]volume={narration_volume}[a]",
                "-map", "[v]", "-map", "[a]",
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
