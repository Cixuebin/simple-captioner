import os
import queue
import threading
import json
import torch
import gradio as gr
from transformers import BitsAndBytesConfig
try:
    from transformers.video_utils import VideoMetadata   # 视频时间轴元数据
except Exception:                                          # 兼容旧版 transformers
    VideoMetadata = None
from qwen_vl_utils import process_vision_info
from PIL import Image
import time
import burner  # 烧录模块：UI 下拉框需要读取 SUBTITLE_POSITIONS 的键名


IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')
VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi', '.webm', '.mkv', ".gif", ".flv")
DEFAULT_PROMPT = "Describe this media."
# 本机部署适配：GTX 1660 只有 6GB 显存，8B 模型 8-bit 需 8.5GB 放不下，
# 因此默认改为 4B + 4-bit（约 3GB，可完整驻留显存）
DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
DEFAULT_QUANT = "4-bit"  # "None" | "8-bit" | "4-bit"
DEFAULT_ATTN = "eager" if os.name == "nt" else "flash_attention_2"

# ---------------------------------------------------------------------------
# 参数记忆（把界面上的设置持久化到 JSON，下次打开自动回填）
# ---------------------------------------------------------------------------
SETTINGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "settings.json")

# 界面上所有希望被记住的控件，以及它们的出厂默认值。
# 新增可持久化的控件时，只要在这里加一行即可。
DEFAULT_SETTINGS = {
    "folder_input": "",
    "tts_engine": "edge（在线·音质好）",
    "keep_audio": True,
    "sub_font_size": 20,
    "sub_position": "底部居中",
    "sub_font_name": "Microsoft YaHei",
    "sub_outline": True,
    "sub_outline_width": 2,
    "sub_opacity": 1.0,
    "orig_volume": 0.5,
    "tts_volume": 1.0,
}

# 控件在界面上的**实际出现顺序**（= 注册 save_settings/reset_settings 时
# inputs/outputs 的顺序）。必须与 _settings_widgets 严格一致，否则参数会写串。
SAVE_ORDER = [
    "folder_input",
    "tts_engine",
    "keep_audio",
    "sub_font_size",
    "sub_position",
    "sub_font_name",
    "sub_outline",
    "sub_outline_width",
    "sub_opacity",
    "orig_volume",
    "tts_volume",
]


def load_settings():
    """读取上次保存的设置；文件不存在或损坏时返回默认值。"""
    data = dict(DEFAULT_SETTINGS)
    try:
        with open(SETTINGS_PATH, encoding="utf-8") as f:
            saved = json.load(f)
        if isinstance(saved, dict):
            # 只接受已知的键，避免旧文件里的脏数据干扰
            for k in DEFAULT_SETTINGS:
                if k in saved:
                    data[k] = saved[k]
    except FileNotFoundError:
        pass
    except Exception as exc:
        print(f"[WARN] 读取 settings.json 失败，使用默认设置：{exc}")
    return data


def save_settings(*values):
    """把界面当前值写入 settings.json，返回回执文本。

    **必须写成普通函数**：Gradio 里 `queue=False` 的事件不支持生成器 ——
    若在这里用 yield，界面只会收到第一条"正在保存"就停住，后面的写盘
    代码永远不会执行（曾经踩过这个坑）。
    写盘是毫秒级操作，不需要中间状态。

    按 SAVE_ORDER 的键名与界面控件一一对应，不用位置硬编码，
    避免以后调整界面顺序时把参数写串。
    """
    # 参数个数对不上说明界面绑定和 SAVE_ORDER 脱节，直接报错而不是写脏数据
    if len(values) != len(SAVE_ORDER):
        return (f"⚠️ 保存失败：参数数量不匹配（收到 {len(values)} 项，"
                f"需要 {len(SAVE_ORDER)} 项），请检查界面绑定顺序。")

    data = dict(zip(SAVE_ORDER, values))
    tmp_path = SETTINGS_PATH + ".tmp"
    try:
        # 先写临时文件再原子替换：避免写到一半中断导致配置损坏
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, SETTINGS_PATH)
        return f"✅ 设置已保存（{time.strftime('%H:%M:%S')}），下次打开自动生效。"
    except Exception as exc:
        # 清理可能残留的临时文件，避免下次误读
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        return f"⚠️ 保存失败：{exc}"


def reset_settings():
    """恢复出厂默认值（仅回填界面，不立刻写盘）。"""
    d = DEFAULT_SETTINGS
    # 输出顺序必须与 _settings_widgets 完全一致
    vals = []
    for key in SAVE_ORDER:
        vals.append(d[key])
    vals.append("↩️ 已恢复默认值（点「💾 保存当前设置」后才会写入文件）")
    return tuple(vals)


# ---------------------------------------------------------------------------
# 中文解说字幕模式
# ---------------------------------------------------------------------------
# 视频：按真实时间轴分段输出中文解说，格式 [起-止] 解说词，可直接做成字幕。
# {duration} / {segments} 在运行时替换成真实时长与建议分段数，
# 避免模型自己"猜"时间、也防止长视频被 token 上限截断。
SUBTITLE_PROMPT_VIDEO = (
    "你是一名专业的视频解说员。请为这段视频撰写**中文解说字幕**，供后期直接添加字幕使用。\n"
    "视频总时长约为 {duration} 秒，请分成约 {segments} 个片段。\n\n"
    "输出要求：\n"
    "1. 用解说员的口吻，语言口语化、生动自然，适合直接朗读；\n"
    "2. 按时间顺序把视频分成上述数量的片段，每个片段单独一行；\n"
    "3. 每行格式严格为：[起-止] 解说词 ，起止时间用秒表示、保留一位小数，"
    "例如：[0.0-3.2] 韩国选手纵身救球！\n"
    "4. 第一段从 0.0 开始，最后一段结束于 {duration}，各片段首尾衔接、覆盖整段视频；\n"
    "5. 每句解说词不超过 20 个字，简洁有力；\n"
    "6. **只输出这些解说行**，不要输出标题、说明、序号、英文或任何多余内容。"
)

# 图片：没有时间轴，改为输出中文解说词（同样适合做旁白/字幕）
SUBTITLE_PROMPT_IMAGE = (
    "你是一名专业的画面解说员。请用**一句简洁生动的中文**解说这张图片的内容，"
    "语言口语化、适合直接朗读，不超过 25 个字。只输出这一句中文，不要任何多余内容。"
)


def suggest_segment_count(duration):
    """按视频时长推算合理分段数：每段约 3 秒，夹在 4~24 段之间。

    作用：短视频不会被拆得过碎，长视频也不会因段数过多而被 max_tokens 截断。
    """
    if not duration or duration <= 0:
        return 8
    n = int(round(duration / 3.0))
    return max(4, min(24, n))


# ---------------------------------------------------------------------------
# SRT 字幕转换
# 设计取舍：模型只输出好写的 `[起-止] 文本`，由代码转成严格的 SRT 语法。
# 这样时间码由程序精确计算（不会多个/少个 0、不会用错分隔符），
# 模型只需专注"看懂画面 + 写好解说词"这一件它擅长的事。
# ---------------------------------------------------------------------------
import re  # noqa: E402  （放在此处是为了就近说明转换逻辑的依赖）

# 匹配形如 [0.0-3.2] 解说词 / 0.0-3.2 解说词 / [0.0 - 3.2] 解说词 的各种写法。
# 兼容要点：
#   - 行首可能有模型自作主张加上的序号（如 "1. " / "1、" / "1)"），需容忍
#   - 连接符可能是半角 (-) 或全角/en-dash/em-dash/波浪号 (– — ～ ~)
_SUBTITLE_LINE_RE = re.compile(
    r"^\s*(?:\d+\s*[.、)．]\s*)?"          # 可选行首序号
    r"[\[【]?\s*"                          # 可选左括号（兼容全角【）
    r"(\d+(?:\.\d+)?)\s*[-–—~～]\s*(\d+(?:\.\d+)?)\s*"
    r"[\]】]?\s*"                          # 可选右括号
    r"(.+?)\s*$"                           # 解说词
)


def seconds_to_srt_timecode(seconds):
    """把秒数转成 SRT 时间码：HH:MM:SS,mmm（例如 3.2 → 00:00:03,200）。

    注意 SRT 用**逗号**分隔毫秒，且时分秒必须补足两位。
    """
    try:
        total_ms = int(round(float(seconds) * 1000))
    except (TypeError, ValueError):
        total_ms = 0
    if total_ms < 0:
        total_ms = 0
    hours, rem = divmod(total_ms, 3600 * 1000)
    minutes, rem = divmod(rem, 60 * 1000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def caption_to_srt(caption):
    """把模型输出的 `[起-止] 解说词` 文本转换成标准 SRT 字幕。

    返回 (srt_text, segment_count)；
    若一行都没解析出来，返回 (None, 0)，调用方应回退到写 .txt。
    """
    if not caption:
        return None, 0

    segments = []
    for raw_line in caption.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = _SUBTITLE_LINE_RE.match(line)
        if not m:
            continue
        start, end, text = m.group(1), m.group(2), m.group(3).strip()
        if not text:
            continue
        # 去掉模型偶尔加上的行首序号（如 "1. " 或 "1、"）
        text = re.sub(r"^\d+\s*[.、)]\s*", "", text).strip()
        if not text:
            continue
        segments.append((float(start), float(end), text))

    if not segments:
        return None, 0

    # 按开始时间排序，并修正明显不合理的时间区间（起 >= 止 或 倒退）
    segments.sort(key=lambda x: x[0])
    fixed = []
    prev_end = 0.0
    for start, end, text in segments:
        if start < prev_end:
            start = prev_end
        if end <= start:
            end = start + 1.0          # 至少给 1 秒，避免零长度字幕
        fixed.append((start, end, text))
        prev_end = end

    blocks = []
    for i, (start, end, text) in enumerate(fixed, start=1):
        blocks.append(
            f"{i}\n"
            f"{seconds_to_srt_timecode(start)} --> {seconds_to_srt_timecode(end)}\n"
            f"{text}"
        )
    # SRT 规范：每个字幕块之间空一行，文件末尾留一个换行
    return "\n\n".join(blocks) + "\n", len(fixed)


PRESETS = {
    "auto": {
        "min_pixels": 256 * 28 * 28,
        "max_pixels": 896 * 28 * 28,
    },
    "fast": {
        "resized_height": 392,
        "resized_width": 392,
    },
    "high": {
        "resized_height": 728,
        "resized_width": 728,
    },
}


AVAILABLE_MODELS = [
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3-VL-4B-Instruct",
    "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "Custom...",
]


# Globals
processor = None
current_model_id = DEFAULT_MODEL_ID
current_quant = DEFAULT_QUANT
model = None
should_abort = False
model_loading = False   # 模型是否正在加载中（用于给用户明确提示，避免误报"文件损坏"）
ui_e = {}

# 本地模型根目录：HuggingFace 不可达时，模型由 ModelScope 下载到此处
MODELS_ROOT = os.environ.get(
    "SC_MODELS_ROOT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"),
)


def resolve_model_path(model_id: str) -> str:
    """把 HuggingFace 风格的模型 ID 映射为本地目录。

    规则：取 ID 最后一段（如 Qwen/Qwen3-VL-8B-Instruct → Qwen3-VL-8B-Instruct），
    若 models/ 下存在同名目录且内含 config.json，则改用该本地路径加载；
    否则原样返回，交给 transformers 走在线下载。
    """
    local_dir = os.path.join(MODELS_ROOT, model_id.split("/")[-1])
    if os.path.isdir(local_dir) and os.path.exists(os.path.join(local_dir, "config.json")):
        print(f"[DEBUG] 命中本地模型目录: {local_dir}")
        return local_dir
    return model_id


def build_bnb_config(quant_choice: str):
    if quant_choice == "8-bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    if quant_choice == "4-bit":
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    return None

def is_qwen35_model(model_id: str) -> bool:
    """Qwen3.5 models use a different class and require thinking-mode handling."""
    return "Qwen3.5" in model_id or "Qwen3_5" in model_id


def pick_model_class(model_id: str):
    from transformers import AutoModelForImageTextToText
    try:
        if is_qwen35_model(model_id):
            from transformers import Qwen3_5ForConditionalGeneration
            return Qwen3_5ForConditionalGeneration
        if "Qwen3-VL" in model_id:
            from transformers import Qwen3VLForConditionalGeneration
            return Qwen3VLForConditionalGeneration
        if "Qwen2.5-VL" in model_id or "Qwen2_5-VL" in model_id:
            from transformers import Qwen2_5_VLForConditionalGeneration
            return Qwen2_5_VLForConditionalGeneration
    except Exception:
        pass
    return AutoModelForImageTextToText

def unload_model():

    print ("[DEBUG] Unload currently loaded model")
    global model, processor
    try:
        del model
    except Exception:
        pass
    try:
        del processor
    except Exception:
        pass
    model = None
    processor = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_selected_model(model_id: str, quant_choice: str, attn_impl: str = DEFAULT_ATTN):
    """
    Loads (or reloads) the model + processor with chosen quantization and attention impl.
    Falls back to 'eager' if flash_attention_2 fails.
    """

    print("[DEBUG] Loading selected model:", model_id)
    global model, processor, current_model_id, current_quant, model_loading

    unload_model()

    # 优先使用本地目录（由 ModelScope 下载），没有则回退到原始在线 ID
    load_path = resolve_model_path(model_id)

    kwargs = {
        "dtype": torch.float16,
        "device_map": "auto",
        "attn_implementation": attn_impl,
    }
    bnb = build_bnb_config(quant_choice)
    if bnb is not None:
        kwargs["quantization_config"] = bnb

    ModelCls = pick_model_class(model_id)
    model_loading = True          # 标记"正在加载"，供 process_folder 前置检查使用
    try:
        try:
            model = ModelCls.from_pretrained(load_path, **kwargs)
        except Exception:
            if attn_impl == "flash_attention_2":
                kwargs["attn_implementation"] = "eager"
                model = ModelCls.from_pretrained(load_path, **kwargs)
            else:
                raise
        from transformers import AutoProcessor as _AP
        processor = _AP.from_pretrained(load_path)
    finally:
        model_loading = False     # 无论成功失败都要复位，否则后续会一直误判为加载中

    # current_model_id 仍保留原始 ID（供 is_qwen35_model 判断），不写入本地路径
    current_model_id = model_id
    current_quant = quant_choice
    return get_model_info()


def toggle_controls(disabled=True):
    updates = {}
    for name, component in ui_e.items():
        if name == "abort_button":
            updates[name] = gr.update(interactive=not disabled)
        else:
            updates[name] = gr.update(interactive=not disabled if component.visible else False)
    return updates


def disable_controls_dict():
    return [toggle_controls(disabled=True)[k] for k in control_keys]

def enable_controls_dict():
    return [toggle_controls(disabled=False)[k] for k in control_keys]

control_keys = [
    "model_dropdown",
    "quant_dropdown",
    "attn_dropdown",
    "load_button",
    "reset_button",
    "start_button",
    "abort_button",
    "folder_input",
    "prompt_input",
    "skip_existing_checkbox",
    "max_tokens_slider",
    "summary_mode",
    "one_sentence_mode",
    "subtitle_mode",
    "retain_preview_checkbox",
    "resolution_mode",
    "status_output",
]


def finish_process():
    updates = enable_controls_dict()
    abort_index = control_keys.index("abort_button")
    updates[abort_index] = gr.update(interactive=False)
    return updates


def abort_process():
    global should_abort
    should_abort = True
    updates = enable_controls_dict()
    abort_index = control_keys.index("abort_button")
    status_index = control_keys.index("status_output")
    updates[abort_index] = gr.update(interactive=False)
    updates[status_index] = gr.update(value="⛔ Aborting process...")
    return updates


def start_process():
    updates = disable_controls_dict()
    abort_index = control_keys.index("abort_button")
    updates[abort_index] = gr.update(interactive=True)
    return updates


def serialize_for_debug(obj):
    if isinstance(obj, dict):
        return {k: serialize_for_debug(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_for_debug(i) for i in obj]
    elif isinstance(obj, Image.Image):
        return f"<Image {obj.size} {obj.mode}>"
    else:
        return obj


def get_model_info():
    global model
    if model is None:
        return "Model not loaded.", "N/A", "N/A", "N/A", "N/A"

    model_name = model.config._name_or_path if hasattr(model.config, "_name_or_path") else "Unknown Model"
    device = "CUDA" if torch.cuda.is_available() else "CPU"

    if torch.cuda.is_available():
        vram_used = f"{torch.cuda.memory_allocated() / 1e9:.2f} GB"
        vram_total = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
    else:
        vram_used, vram_total = "N/A", "N/A"

    dtype = str(next(model.parameters()).dtype)
    return model_name, device, f"{vram_used} / {vram_total}", dtype, str(model.config)


print("[DEBUG] Cuda available:", torch.cuda.is_available())

def generate_caption(media_path, prompt, max_tokens, summary_mode=False, one_sentence_mode=False,
                     resolution_mode="auto", subtitle_mode=False, progress_callback=None):
    """生成一个媒体文件的描述/字幕。

    progress_callback: 可选回调，签名 (stage: str, detail: str) -> None。
        用于向界面上报当前阶段，让用户在处理长视频时能看到进度。
        阶段：decode（解码/抽帧）、build（构建输入）、generate（模型推理）、decode_text（整理输出）
    """
    global processor, model
    assert model is not None, "Model must be loaded before generating captions."
    assert processor is not None, "Processor must be loaded before generating captions."

    def report(stage, detail=""):
        """安全上报进度：回调异常不能影响主流程。"""
        if progress_callback is not None:
            try:
                progress_callback(stage, detail)
            except Exception:
                pass

    ext = os.path.splitext(media_path)[-1].lower()
    is_video = ext in VIDEO_EXTENSIONS

    # ---- 解码媒体：视频需要先解码，因为解说模式要用真实时长构建提示词 ----
    video_meta = None
    video_duration = None
    if is_video:
        report("decode", "正在解码视频并抽取关键帧 ...")
        # 用 PyAV 解码成帧列表（规避 torchvision 0.29 已移除 read_video 的坑）
        media_data, sample_fps, video_duration = decode_video_frames(media_path)
        report("decode", f"已抽取 {len(media_data)} 帧"
                         + (f"（时长 {video_duration:.1f}s）" if video_duration else ""))
        # 把真实时间轴告诉 processor，否则它会按默认 24fps 走，时间戳全错
        if VideoMetadata is not None:
            video_meta = [VideoMetadata(
                total_num_frames=len(media_data),
                fps=sample_fps,
                frames_indices=list(range(len(media_data))),
                duration=len(media_data) / sample_fps,
                width=media_data[0].width,
                height=media_data[0].height,
            )]
    else:
        report("decode", "正在读取图片 ...")
        media_data = Image.open(media_path).convert("RGB")

    # ---- 构建最终提示词 ----
    if subtitle_mode:
        # 中文解说字幕模式：忽略用户自定义 prompt，用专用模板
        if is_video:
            dur_text = f"{video_duration:.1f}" if video_duration else "未知"
            seg = suggest_segment_count(video_duration)
            final_prompt = SUBTITLE_PROMPT_VIDEO.format(duration=dur_text, segments=seg)
        else:
            final_prompt = SUBTITLE_PROMPT_IMAGE
    else:
        final_prompt = prompt
        # Edit custom prompts here if you want
        if summary_mode and one_sentence_mode:
            final_prompt += " Give a one-sentence summary of the scene."
        elif summary_mode:
            final_prompt += " Give a short summary of the scene."
        elif one_sentence_mode:
            final_prompt += " Describe this image in one sentence."

    content_type = "video" if is_video else "image"
    content_block = {"type": content_type, content_type: media_data}

    if not is_video:

        if resolution_mode == "auto":
            content_block["min_pixels"] = 256 * 28 * 28
            content_block["max_pixels"] = 896 * 28 * 28
        elif resolution_mode == "auto_high":
            content_block["min_pixels"] = 256 * 28 * 28
            content_block["max_pixels"] = 1280 * 28 * 28
        elif resolution_mode == "fast":
            content_block["resized_height"] = 392
            content_block["resized_width"] = 392
        elif resolution_mode == "high":
            content_block["resized_height"] = 728
            content_block["resized_width"] = 728


    report("build", "正在构建模型输入 ...")
    messages = [
        {
            "role": "user",
            "content": [
                content_block,
                {"type": "text", "text": final_prompt},
            ],
        }
    ]

    qwen35 = is_qwen35_model(current_model_id)
    if qwen35:
        messages.append({"role": "assistant", "content": "<think>\n\n</think>\n\n"})

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=not qwen35,
        continue_final_message=qwen35,
    )

    vision_info = process_vision_info(messages)

    assert isinstance(vision_info, tuple), "process_vision_info did not return a tuple"

    if len(vision_info) == 3:
        image_inputs, video_inputs, _ = vision_info
    elif len(vision_info) == 2:
        image_inputs, video_inputs = vision_info # type: ignore
    else:
        raise ValueError(f"Error: Expected 2 or 3 values from process_vision_info, but got {len(vision_info)}.")

    proc_kwargs = {
        "text": [text],
        "images": image_inputs,
        "videos": video_inputs,
        "padding": True,
        "return_tensors": "pt",
    }
    if video_meta is not None:
        # 帧已由我们自己抽好，必须告诉 processor：
        #   do_sample_frames=False —— 否则它会按 fps=24 重新采样，把 32 帧砍到 2 帧
        #   video_metadata        —— 提供真实时间轴，保证 <x.x seconds> 时间戳正确
        #   cap_pixels_per_frame  —— 对齐官方 qwen-vl-utils 的每帧像素上限，防高分辨率视频爆 token
        proc_kwargs.update(
            do_sample_frames=False,
            cap_pixels_per_frame=True,
            video_metadata=video_meta,
        )

    inputs = processor(**proc_kwargs).to("cuda")


    report("generate", "正在运行模型推理 ...")
    # 关键：model.generate() 是同步阻塞调用，长视频可能跑 30~60 秒。
    # 放进后台线程执行，主线程定期上报心跳，这样界面上能看到"已用时"持续增长，
    # 而不是整段卡死毫无反馈。
    holder = {"ids": None, "err": None, "done": False}

    def _run_generate():
        try:
            with torch.no_grad():
                holder["ids"] = model.generate(**inputs, max_new_tokens=max_tokens)
        except Exception as exc:          # 线程内异常要带回主线程再抛
            holder["err"] = exc
        finally:
            holder["done"] = True

    worker = threading.Thread(target=_run_generate, daemon=True)
    worker.start()

    t_gen = time.time()
    while not holder["done"]:
        worker.join(timeout=2.0)          # 每 2 秒醒一次，上报一次心跳
        if not holder["done"]:
            report("generate", f"正在运行模型推理 ... 已用时 {time.time() - t_gen:.0f}s")

    if holder["err"] is not None:
        raise holder["err"]

    generated_ids = holder["ids"]
    report("generate", f"推理完成，用时 {time.time() - t_gen:.1f}s")

    report("decode_text", "正在整理输出文本 ...")
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    caption = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    if qwen35 and "<think>" in caption:
        import re
        caption = re.sub(r"<think>.*?</think>\s*", "", caption, flags=re.DOTALL)

    return caption.strip()


# ---- 视频解码（改用 PyAV，绕开 torchvision 移除 read_video 的问题）----
# 关于帧数上限：实测 8B 模型在 eager attention 下，显存占用随帧数近似线性增长
# （960x540：8帧 18GB / 16帧 22GB / 32帧 35GB），帧数再多就会打满 48GB 显存，
# 触发内存交换导致推理"假死"（界面无响应、无报错）。故上限保守设在 24 帧。
VIDEO_SAMPLE_FPS = 2.0         # 每秒抽取多少帧喂给模型
VIDEO_MIN_FRAMES = 4           # 最少抽帧数（短视频也要有足够信息量）
VIDEO_MAX_FRAMES = 24          # 最多抽帧数（24帧约需 26GB 显存，安全余量充足）
VIDEO_MAX_DECODE_SECONDS = 60  # 单个视频最多解码这么多秒，超出部分直接不抽
VIDEO_MAX_FRAME_SIDE = 640     # 抽帧后最长边上限，控制每帧像素数


def decode_video_frames(video_path, target_fps=VIDEO_SAMPLE_FPS,
                        max_frames=VIDEO_MAX_FRAMES, min_frames=VIDEO_MIN_FRAMES,
                        max_decode_seconds=VIDEO_MAX_DECODE_SECONDS,
                        max_side=VIDEO_MAX_FRAME_SIDE):
    """用 PyAV 把视频解码成 PIL 帧列表，返回 (frames, 实际采样帧率, 视频真实时长秒)。

    背景：qwen_vl_utils 读取视频时按 torchcodec → decord → torchvision 的顺序选后端。
    本机 decord 没有 Windows 版、torchcodec 未装，只能回落 torchvision；
    而 torchvision 0.19+ 已移除 io.read_video，于是所有视频都会 AttributeError 失败。
    这里自行解码成帧列表，再交给 process_vision_info 的"帧列表"分支处理，
    完全不依赖上述三个后端（PyAV 自带 FFmpeg 库，wheel 装完即可用）。

    抽帧策略（关键：解码耗时 ≈ 需要解码的帧数，4K 约 11ms/帧）：
      - 有效分析时长 = min(视频时长, max_decode_seconds)；
        抽帧时间戳只铺在这段区间内，因此超长视频不会为了"均匀覆盖全片"
        而把整段视频解码一遍（16 分钟 4K 实测从 327s 降到约 30s）。
      - 目标帧数 = 有效时长 × target_fps，夹在 [min_frames, max_frames] 之间。
      - 按**时间戳**定位（而非帧序号步进），对手机常见的可变帧率(VFR)视频同样准确。
    """
    import av  # 延迟导入，避免拖慢仅处理图片时的启动速度

    container = av.open(video_path)
    try:
        if not container.streams.video:
            raise RuntimeError(f"文件里没有视频流: {video_path}")
        stream = container.streams.video[0]
        tb = float(stream.time_base)

        # 视频时长（秒）：优先用容器时长，退化到流时长
        duration = None
        if container.duration:
            duration = container.duration / 1_000_000
        elif stream.duration:
            duration = float(stream.duration) * tb

        # 只解码前 max_decode_seconds 秒，够用即可（模型本身也只能看几十帧）
        scan_duration = min(duration, max_decode_seconds) if duration else None

        if scan_duration and scan_duration > 0:
            n_target = max(min_frames, min(max_frames, int(round(scan_duration * target_fps))))
            # 每帧目标时间戳：在扫描区间内均匀分布（首帧 0，末帧贴近区间末尾）
            targets = [i * scan_duration / n_target for i in range(n_target)]
            interval = scan_duration / n_target
        else:
            # 时长未知：按固定间隔顺推，抽满上限为止
            n_target = max_frames
            targets = None
            interval = 1.0 / target_fps

        frames = []
        ti = 0                      # targets 游标
        next_t = (targets[0] if targets else 0.0)
        last_t = 0.0
        for frame in container.decode(stream):
            # 用帧时间戳定位；个别帧缺 pts 时按间隔顺推
            t = float(frame.pts) * tb if frame.pts is not None else last_t + interval
            last_t = t
            if t + 1e-9 < next_t:
                continue            # 还没到下一个采样点，跳过（不转 PIL，省内存与时间）
            img = frame.to_image().convert("RGB")
            # 主动缩小到目标尺寸：4K 帧若原样进入视觉编码器，
            # 会因 token 数暴增而打满显存（实测 53 帧 1080p 直接卡死）。
            if max(img.size) > max_side:
                scale = max_side / max(img.size)
                img = img.resize(
                    (max(2, int(img.width * scale)), max(2, int(img.height * scale))),
                    Image.LANCZOS,
                )
            frames.append(img)
            ti += 1
            if targets is not None:
                if ti >= len(targets):
                    break
                next_t = targets[ti]
            else:
                next_t += interval
                if len(frames) >= n_target:
                    break
    finally:
        container.close()

    if not frames:
        raise RuntimeError(f"视频解码后没有任何帧: {video_path}")

    # 实际采样帧率：以真正被扫描的区间为基准，保证模型拿到的时间轴正确
    base = scan_duration if scan_duration else duration
    sample_fps = (len(frames) / base) if (base and base > 0) else target_fps
    # 真实时长一并返回：生成"解说字幕"时要把精确时长写进提示词，
    # 否则模型只能靠猜，时间标记会和真实视频对不上。
    return frames, sample_fps, (duration if duration and duration > 0 else None)


def is_image_file(filename):
    return filename.lower().endswith(IMAGE_EXTENSIONS)


def is_video_file(filename):
    return filename.lower().endswith(VIDEO_EXTENSIONS)


def build_final_prompt(user_prompt, summary, one_sentence):
    parts = [user_prompt.strip()]
    if summary:
        parts.append("Please provide a short summary.")
    if one_sentence:
        parts.append("Keep the description to one sentence.")
    return " ".join(parts)


def process_folder(folder_path, prompt, skip_existing, max_tokens, summary_mode, one_sentence_mode, retain_preview, resolution_mode, subtitle_mode=False):

    global should_abort, current_model_id, current_quant

    processed_media = 0
    skipped_media = 0
    failed_media = 0
    first_error = None
    last_media_to_show = None
    last_caption = ""
    last_media_name_markdown = ""
    elapsed_str = ""

    # ---- 前置检查：模型未就绪时给出明确提示，避免逐文件误报"文件损坏" ----
    # 判据以"模型是否真的在内存里"为准（model is None），
    # 布尔标志 model_loading 只用于区分"正在加载"与"从未加载"两种情况。
    if model is None or processor is None:
        control_updates = finish_process()
        if model_loading:
            msg = ("⏳ 模型正在加载中，请稍候……"
                   "（8B 模型约需 30~60 秒，加载完成前请勿点击开始处理）")
        else:
            msg = ("⚠️ 模型尚未加载。请先点击【📦 Load / Reload Model】按钮，"
                   "等状态栏显示 Ready 后再点击【🚀 Start Processing】。")
        yield (
            msg,
            None,
            None,
            msg,
            0,
            "",
            *control_updates
        )
        return

    print("[DEBUG] starting folder processing...:")
    print("[DEBUG] using model: ", current_model_id)
    print("[DEBUG] model quantization: ", current_quant)
    print("[DEBUG] folder_path: ", folder_path)
    print("[DEBUG] prompt: ", prompt)
    print("[DEBUG] skip_existing: ", skip_existing)
    print("[DEBUG] max_tokens: ", max_tokens)
    print("[DEBUG] summary_mode: ", summary_mode)
    print("[DEBUG] one_sentence_mode: ", one_sentence_mode)
    print("[DEBUG] retain_preview: ", retain_preview)
    print("[DEBUG] resolution_mode: ", resolution_mode)
    print("[DEBUG] subtitle_mode: ", subtitle_mode)
    print("[DEBUG] should_abort: ", should_abort)

    if not folder_path.strip():
        control_updates = finish_process()
        yield (
            "⚠️ Please enter a valid folder path.",
            None,
            None,
            "No media to process.",
            0,
            "",
            *control_updates
        )
        return

    if not os.path.exists(folder_path):
        control_updates = finish_process()
        yield (
            f"❌ Folder not found: {folder_path}",
            None,
            None,
            "No media to process.",
            0,
            "",
            *control_updates
        )
        return

    media_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:

            if is_image_file(file) or is_video_file(file):
                media_files.append(os.path.join(root, file))

    total_media = len(media_files)

    if total_media == 0:
        control_updates = finish_process()
        yield (
            "📂 No media found in the folder or subfolders.",
            None,
            None,
            "No media to process.",
            0,
            "",
            *control_updates
        )
        return

    start_time = time.time()

    for idx, media_path in enumerate(media_files):
        if should_abort:
            should_abort = False
            status_index = control_keys.index("status_output")
            abort_index = control_keys.index("abort_button")
            control_updates = enable_controls_dict()
            control_updates[status_index] = gr.update(value="⛔ Aborted by user.")
            control_updates[abort_index] = gr.update(interactive=False)

            yield (
                "⛔ Aborted by user.",
                None,
                None,
                "Aborted.",
                0,
                "",
                *control_updates
            )
            return
        try:
            # 输出文件：
            #   字幕解说模式 + 视频 → 同名 .srt（标准字幕文件，播放器/剪辑软件可直接识别）
            #   其它情况（图片 / 关闭字幕模式） → 同名 .txt
            base_name = os.path.splitext(os.path.basename(media_path))[0]
            out_dir = os.path.dirname(media_path)
            txt_path = os.path.join(out_dir, base_name + ".txt")
            srt_path = os.path.join(out_dir, base_name + ".srt")
            use_srt = subtitle_mode and is_video_file(media_path)
            primary_path = srt_path if use_srt else txt_path

            rel_path = os.path.relpath(media_path, folder_path)
            media_name_markdown = f"**File:** `{rel_path}`"

            if skip_existing and os.path.exists(primary_path):
                progress = int(((idx + 1) / total_media) * 100)
                elapsed = int(time.time() - start_time)
                elapsed_str = f"{elapsed//60:02d}:{elapsed%60:02d}"
                skipped_media += 1

                control_updates = start_process()
                yield (
                    f"⏭️ Skipped {idx+1}/{total_media}: {rel_path} (already captioned)",
                    last_media_to_show if retain_preview else None,
                    last_media_name_markdown if retain_preview else None,
                    last_caption if retain_preview else "Skipped (already captioned)",
                    progress,
                    elapsed_str,
                    *control_updates
                )
                continue

            # ---- 处理前先上报一次，让用户立刻看到"正在处理第 N/M 个" ----
            # 这是修复"看不到进度"的关键：原版是等 generate 跑完才 yield，
            # 长视频时界面会几十秒毫无反馈。这里先 yield 一次，界面立即有响应。
            control_updates = start_process()
            progress = int((idx / total_media) * 100)
            elapsed = int(time.time() - start_time)
            elapsed_str = f"{elapsed//60:02d}:{elapsed%60:02d}"
            yield (
                f"⏳ [{idx+1}/{total_media}] 开始处理: {rel_path}",
                last_media_to_show if retain_preview else None,
                media_name_markdown,
                "⏳ 处理中 ...",
                progress,
                elapsed_str,
                *control_updates
            )

            # ---- 用队列接收 generate_caption 的阶段回调，边跑边刷新界面 ----
            # generate_caption 在工作线程里跑，回调把状态写进队列；
            # 主线程轮询队列，一有新状态就 yield，实现细粒度进度更新。
            stage_q = queue.Queue()
            result_holder = {"caption": None, "error": None, "done": False}

            # 各阶段在"当前文件内部"的完成占比，用于让进度条平滑推进：
            # 只有一个视频时，若只按文件数算会一直停在 0%，用户仍会觉得没进展。
            STAGE_WEIGHT = {
                "decode": 0.10,
                "build": 0.20,
                "generate": 0.85,   # 推理是最耗时的一步，占据主要权重
                "decode_text": 1.0,
            }
            stage_ratio = 0.0

            def _overall_progress():
                """把"已完成文件数 + 当前文件内部进度"折算成 0~100 的总进度。"""
                within = (idx + stage_ratio) / max(1, total_media)
                return max(0, min(100, int(within * 100)))

            def _on_progress(stage, detail):
                # 该回调运行在工作线程中，只做入队，不做任何 UI 操作
                stage_q.put((stage, detail))

            def _run_caption():
                try:
                    # 解说字幕模式下，多段解说需要更长输出，自动抬高上限（至少 512 tokens）
                    eff_tokens = max(max_tokens, 512) if subtitle_mode else max_tokens
                    result_holder["caption"] = generate_caption(
                        media_path, prompt, eff_tokens, summary_mode,
                        one_sentence_mode, resolution_mode, subtitle_mode,
                        progress_callback=_on_progress,
                    )
                except Exception as exc:
                    result_holder["error"] = exc
                finally:
                    result_holder["done"] = True
                    stage_q.put(("__done__", ""))

            cap_worker = threading.Thread(target=_run_caption, daemon=True)
            cap_worker.start()

            t_item = time.time()
            while not result_holder["done"]:
                try:
                    stage, detail = stage_q.get(timeout=1.0)
                except queue.Empty:
                    # 没有新阶段消息：上报一次心跳，让用户看到计时在走
                    elapsed_now = time.time() - t_item
                    elapsed = int(time.time() - start_time)
                    elapsed_str = f"{elapsed//60:02d}:{elapsed%60:02d}"
                    yield (
                        f"⚙️ [{idx+1}/{total_media}] 处理中: {rel_path} "
                        f"（当前文件已用 {elapsed_now:.0f}s）",
                        last_media_to_show if retain_preview else None,
                        media_name_markdown,
                        f"⚙️ 处理中 ... 当前文件已用 {elapsed_now:.0f}s",
                        _overall_progress(),
                        elapsed_str,
                        *control_updates
                    )
                    continue

                if stage == "__done__":
                    break

                # 记录当前文件内部进度，让总进度条平滑推进而非长时间停滞
                stage_ratio = STAGE_WEIGHT.get(stage, stage_ratio)
                elapsed = int(time.time() - start_time)
                elapsed_str = f"{elapsed//60:02d}:{elapsed%60:02d}"
                yield (
                    f"⚙️ [{idx+1}/{total_media}] {detail}",
                    last_media_to_show if retain_preview else None,
                    media_name_markdown,
                    f"⚙️ {detail}",
                    _overall_progress(),
                    elapsed_str,
                    *control_updates
                )

            # 等线程真正结束，取出结果或抛出异常
            cap_worker.join()
            if result_holder["error"] is not None:
                raise result_holder["error"]
            caption = result_holder["caption"]

            # Only generate previews for images
            if is_image_file(media_path):
                media_to_show = Image.open(media_path)
            else:
                media_to_show = None

            # ---- 写文件 ----
            # 字幕模式下的视频：把 `[起-止] 文本` 转成标准 SRT 写 .srt。
            # 若一行都没解析出来（模型没按格式输出），回退写 .txt，避免产生空文件。
            written_path = txt_path
            extra_note = ""
            if use_srt:
                srt_text, seg_count = caption_to_srt(caption)
                if srt_text:
                    with open(srt_path, "w", encoding="utf-8") as f:
                        f.write(srt_text)
                    written_path = srt_path
                    extra_note = f" · 已生成 SRT（{seg_count} 条字幕）"
                else:
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(caption)
                    written_path = txt_path
                    extra_note = " · ⚠️ 未识别到时间轴，已回退为 .txt"
                    print(f"[WARN] SRT 转换失败，回退写 .txt: {txt_path}")
            else:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(caption)

            progress = int(((idx + 1) / total_media) * 100)
            elapsed = int(time.time() - start_time)
            elapsed_str = f"{elapsed//60:02d}:{elapsed%60:02d}"

            last_media_to_show = media_to_show
            last_caption = caption
            last_media_name_markdown = media_name_markdown
            processed_media += 1
            control_updates = start_process()
            icon = "🎬" if written_path.endswith(".srt") else "🖼️"
            yield (
                f"{icon} [{idx+1}/{total_media}] 完成: {rel_path} "
                f"→ {os.path.basename(written_path)}{extra_note}",
                media_to_show,
                media_name_markdown,
                caption,
                progress,
                elapsed_str,
                *control_updates
            )

        except Exception as e:
            failed_media += 1
            error_text = f"{type(e).__name__}: {e}"
            if first_error is None:
                first_error = error_text      # 记住第一个真实错误，最后汇总展示
            print(f"[ERROR] Failed processing file {media_path}: {e}")
            control_updates = start_process()
            yield (
                f"⚠️ Error processing {rel_path}: {error_text}",
                None,
                None,
                "Error in captioning.",
                0,
                elapsed_str,
                *control_updates
            )

    end_time = time.time()
    total_secs = end_time - start_time
    total_time = f"{total_secs:.2f}"
    elapsed_str = f"{int(total_secs)//60:02d}:{int(total_secs)%60:02d}"

    control_updates = finish_process()
    # 汇总消息：中文 + 信息完整（总数/成功/跳过/失败/总耗时）
    head = f"✅ 处理完成！共 {total_media} 个文件"
    detail = (f"成功 {processed_media}，跳过 {skipped_media}，失败 {failed_media}，"
              f"总耗时 {elapsed_str}")
    if failed_media > 0:
        # 有失败时，把真实错误原因带出来，而不是笼统地说"文件损坏"
        summary = f"{head}；{detail}。首个错误 -> {first_error}"
    else:
        summary = f"{head}；{detail}。"

    # 完成时进度条置为 100，避免停在未满状态让用户误以为还没跑完
    yield (
        summary,
        last_media_to_show,
        last_media_name_markdown,
        last_caption,
        100,
        elapsed_str,
        *control_updates
    )


def reset_prompt():
    return DEFAULT_PROMPT


css = """
.generating {
    border: none;
}
"""


with gr.Blocks() as iface: # type: ignore

    gr.Markdown("# Simple Captioner")
    gr.Markdown("A simple media caption generator for images and video using **[Qwen2.5/3/3.5 VL Instruct](https://huggingface.co/Qwen/)**")
    gr.Markdown("Supported image formats: png, jpg, jpeg, bmp, gif, webp")
    gr.Markdown("Supported video formats: mp4, mov, avi, webm, mkv, gif, flv")
    gr.Markdown("Written by [Olli S.](https://github.com/o-l-l-i)")

    gr.Markdown(
        "> **⚠️ Before processing:** click **📦 Load / Reload Model** and wait until the Status bar "
        "shows **\"✅ Loaded ... Ready to process.\"** (takes ~30–60s). "
        "If you click **Start Processing** before that, it will fail with *\"No model loaded\"*."
    )


    with gr.Accordion("⚙️ Model Settings", open=True):
        ui_e["model_dropdown"] = model_dropdown = gr.Dropdown(
            label="Model",
            choices=AVAILABLE_MODELS,
            value=DEFAULT_MODEL_ID,
            allow_custom_value=False,
            interactive=True,
            info="Pick a model to use for captioning."
        )
        custom_model_box = gr.Textbox(
            label="Custom Model ID (Hugging Face)",
            placeholder="e.g. Qwen/Qwen3-VL-4B-Instruct or your-org/my-qwen3-checkpoint",
            visible=False,
        )
        ui_e["quant_dropdown"] = quant_dropdown = gr.Radio(
            label="Quantization",
            choices=["None", "8-bit", "4-bit"],
            value=DEFAULT_QUANT,
            interactive=True,
            info="Lower-bit quantization reduces VRAM, may slightly affect quality."
        )
        ui_e["attn_dropdown"] = attn_dropdown = gr.Radio(
            label="Attention Implementation",
            choices=["flash_attention_2", "eager"],
            value=DEFAULT_ATTN,
            interactive=True,
            info="If FlashAttention isn't installed/working, choose 'eager'. Auto-fallback on load."
        )
        ui_e["load_button"] = gr.Button("📦 Load / Reload Model")

    def _toggle_custom(choice):
        return gr.update(visible=(choice == "Custom…"))

    model_dropdown.change(_toggle_custom, inputs=[model_dropdown], outputs=[custom_model_box])

    def _ui_load_model(sel, custom_id, quant, attn):
        model_id = custom_id.strip() if sel == "Custom..." and custom_id and custom_id.strip() else sel
        # 先立刻回一条"正在加载"，避免用户在加载期间误以为卡住而反复点击
        loading_msg = f"⏳ Loading '{model_id}' ({quant}) ... please wait ~30-60s, do NOT click Start Processing yet."
        yield loading_msg, "loading...", "N/A", "N/A", "N/A", "N/A"
        name, device, vram, dtype, cfg = load_selected_model(model_id, quant, attn)
        status = f"✅ Loaded '{model_id}' with {quant} quantization ({attn}). Ready to process."
        print ("[DEBUG]", status)
        yield status, name, device, vram, dtype, cfg
    with gr.Accordion("⚙️ Model Information", open=False):
        model_name_display = gr.Textbox(label="Model Name", interactive=False)
        device_display = gr.Textbox(label="Device", interactive=False)
        vram_display = gr.Textbox(label="VRAM Usage", interactive=False)
        dtype_display = gr.Textbox(label="Torch Dtype", interactive=False)
        config_display = gr.Textbox(label="Model Config", interactive=False, lines=4)

    # 读取上次保存的设置，供后续所有控件初始值使用。
    # 必须放在第一个需要回填的控件（文件夹路径）之前 —— 之前写在烧录区，
    # 导致 folder_input 构造时还没读到配置，路径永远回填不了。
    _saved = load_settings()

    with gr.Row():
        ui_e["folder_input"] = gr.Textbox(
            label="📁 Folder Path",
            value=_saved["folder_input"],
            placeholder="e.g. C:\\Users\\you\\Pictures\\input_images",
        )
        ui_e["prompt_input"] = gr.Textbox(label="Custom Prompt", value=DEFAULT_PROMPT)

    ui_e["skip_existing_checkbox"] = gr.Checkbox(
        label="Skip already captioned media (同名 .srt / .txt 已存在则跳过)", value=False
    )


    with gr.Row():
        gr.Markdown("### Prompt Controls")
        gr.Markdown("""
        - **Summary Mode**: Asks the model to summarize the media content briefly.
        - **One-Sentence Mode**: Instructs the model to keep the caption to a single concise sentence.
        - **字幕解说模式 (Subtitle)**: 输出中文解说字幕；视频生成标准 `.srt` 字幕文件（序号 + 时间码 `00:00:00,000 --> 00:00:03,200`），播放器/剪辑软件可直接加载。
        """)
        ui_e["summary_mode"] = gr.Checkbox(label="Summary Mode", value=False)
        ui_e["one_sentence_mode"] = gr.Checkbox(label="One-Sentence Mode", value=False)
        ui_e["subtitle_mode"] = gr.Checkbox(
            label="🎬 字幕解说模式（中文·视频导出 .srt）",
            value=True,
            info="勾选后：视频 → 同名 .srt 标准字幕；图片 → 同名 .txt 单句解说",
        )


    prompt_preview = gr.Textbox(
        label="Final Prompt Preview",
        lines=2,
        interactive=False
    )

    with gr.Row():
        ui_e["max_tokens_slider"] = gr.Slider(label="🧾 Max Tokens", minimum=32, maximum=1024, value=512, step=16,
                                             info="字幕解说模式会按需自动抬高（≥512）")
        ui_e["resolution_mode"] = gr.Dropdown(
            label="Image Resolution",
            choices=["auto", "auto_high", "fast", "high"],
            value="auto",
            info="Choose the resolution mode for visual input."
        )

    with gr.Row():
        ui_e["reset_button"] = gr.Button("🔄 Reset to Default Prompt")
        ui_e["start_button"] = gr.Button("🚀 Start Processing", interactive=True)
        ui_e["abort_button"] = gr.Button("⛔ Abort", interactive=False)

    ui_e["status_output"] = gr.Textbox(label="Status", interactive=False)
    progress_bar = gr.Slider(
        minimum=0, maximum=100, value=0, step=1,
        label="📊 总进度 (%)",
        info="按文件数统计。单个文件处理中的实时步骤请看上方 Status。",
        interactive=False,
    )
    time_display = gr.Textbox(label="⏱️ 总耗时 (mm:ss)", interactive=False)

    # ================= 🎥 烧录字幕 + 语音旁白 =================
    # 独立于"开始处理"：对文件夹里已有 .srt 的视频逐个烧录硬字幕，
    # 并把解说词用 TTS 合成旁白混进音轨。字幕样式与音量均可在下方调节。
    gr.Markdown("### 🎥 合成旁白视频（烧录字幕 + TTS 解说）")
    gr.Markdown(
        "> 对上方文件夹内**已有 .srt** 的视频逐个处理，输出 `_旁白版.mp4`。\n"
        "> 需要 ffmpeg（自动探测；也可用环境变量 `SC_FFMPEG` 指定路径）。"
    )

    # 注：_saved 已在文件夹输入框之前读取（见上方），这里不再重复 load

    with gr.Row():
        ui_e["tts_engine"] = gr.Radio(
            label="🗣️ 旁白语音引擎",
            choices=["edge（在线·音质好）", "sapi（离线·机械音）"],
            value=_saved["tts_engine"],
            info="edge 走微软在线服务，失败会自动回退到 Windows 离线语音",
        )
        ui_e["keep_audio"] = gr.Checkbox(
            label="保留视频原声", value=_saved["keep_audio"],
            info="取消勾选则只保留 TTS 解说音轨",
        )

    with gr.Accordion("🎨 字幕样式设置", open=True):
        with gr.Row():
            ui_e["sub_font_size"] = gr.Slider(
                label="🔠 字幕大小",
                minimum=8, maximum=96, value=_saved["sub_font_size"], step=1,
                info="ASS 字号。1080p 竖屏建议 20~32；横屏 1080p 建议 28~44",
            )
            ui_e["sub_position"] = gr.Dropdown(
                label="📍 字幕位置",
                choices=list(burner.SUBTITLE_POSITIONS.keys()),
                value=_saved["sub_position"],
            )
        ui_e["sub_font_name"] = gr.Dropdown(
            label="🔤 字体",
            choices=[
                "Microsoft YaHei", "Microsoft YaHei UI", "SimHei",
                "SimSun", "KaiTi", "Source Han Sans CN", "Arial",
            ],
            value=_saved["sub_font_name"],
            allow_custom_value=True,
        )
        with gr.Row():
            ui_e["sub_outline"] = gr.Checkbox(
                label="🖊️ 字幕描边（黑边）",
                value=_saved["sub_outline"],
                info="取消勾选则只有纯色文字，无黑边",
            )
            ui_e["sub_outline_width"] = gr.Slider(
                label="🖊️ 描边粗细",
                minimum=0, maximum=4, value=_saved["sub_outline_width"], step=1,
                info="仅在上方勾选描边时生效",
            )
        ui_e["sub_opacity"] = gr.Slider(
            label="🫥 字幕不透明度",
            minimum=0.0, maximum=1.0, value=_saved["sub_opacity"], step=0.05,
            info="1.0 = 完全不透明；0.6 左右可让字幕透出画面",
        )

    with gr.Accordion("🔊 音量设置", open=True):
        with gr.Row():
            ui_e["orig_volume"] = gr.Slider(
                label="🎬 视频原声音量",
                minimum=0.0, maximum=2.0, value=_saved["orig_volume"], step=0.05,
                info="0 = 静音，0.5 = 压低一半（默认），1.0 = 原始音量",
            )
            ui_e["tts_volume"] = gr.Slider(
                label="🗣️ TTS 解说的音量",
                minimum=0.0, maximum=3.0, value=_saved["tts_volume"], step=0.05,
                info="1.0 = 原始大小；觉得解说太小声就调大",
            )

    # 参数记忆：保存 / 恢复默认 两个按钮 + 一个回执框
    with gr.Row():
        ui_e["save_settings_button"] = gr.Button("💾 保存当前设置", size="sm")
        ui_e["reset_settings_button"] = gr.Button("↩️ 恢复默认值", size="sm")
    settings_status = gr.Textbox(
        label="设置状态", interactive=False, max_lines=1,
        placeholder="调好参数后点「保存当前设置」，下次打开会自动回填",
    )

    ui_e["burn_button"] = gr.Button("🎥 开始烧录字幕+旁白")
    burn_status = gr.Textbox(label="烧录状态", interactive=False)

    def _burn_folder(folder, engine_choice, keep_audio, font_size,
                     font_name, sub_position, orig_vol, tts_vol,
                     outline_enabled, outline_width, text_opacity):
        """遍历文件夹，把已有 .srt 的视频烧录成带旁白的成片。生成器。"""
        import burner  # 延迟导入，加快启动

        engine = "sapi" if engine_choice.startswith("sapi") else "edge"
        if not folder or not os.path.isdir(folder):
            yield "⚠️ 请先在上方填一个有效的文件夹路径。"
            return

        # 收集待处理对 (视频, srt)
        pairs = []
        for root, _dirs, files in os.walk(folder):
            for fn in files:
                if not burner.is_video(fn):
                    continue
                srt = os.path.join(root, os.path.splitext(fn)[0] + ".srt")
                if os.path.exists(srt):
                    pairs.append((os.path.join(root, fn), srt))
        if not pairs:
            yield "⚠️ 文件夹里没有找到「视频 + 同名 .srt」的组合。请先跑一次字幕处理。"
            return

        yield f"⏳ 共找到 {len(pairs)} 个待烧录视频，开始处理 ..."
        ok, fail = 0, 0
        for i, (video, srt) in enumerate(pairs):
            out = os.path.splitext(video)[0] + "_旁白版.mp4"
            q = queue.Queue()

            def _cb(stage, detail, _q=q):
                _q.put(f"{stage}:{detail}")

            def _run(_v=video, _s=srt, _o=out):
                try:
                    burner.burn_subtitles_with_tts(
                        _v, _s, _o, engine=engine,
                        # 字幕样式（来自 Web 端「字幕样式设置」）
                        font_name=font_name,
                        font_size=font_size,
                        subtitle_position=sub_position,
                        outline_enabled=outline_enabled,
                        outline_width=outline_width,
                        text_opacity=text_opacity,
                        # 音量（来自 Web 端「音量设置」）
                        original_volume=orig_vol,
                        narration_volume=tts_vol,
                        keep_original_audio=keep_audio,
                        progress_cb=_cb,
                    )
                    q.put("__ok__")
                except Exception as exc:
                    q.put(f"__err__ {exc}")

            worker = threading.Thread(target=_run, daemon=True)
            worker.start()
            t0 = time.time()
            while True:
                try:
                    msg = q.get(timeout=2.0)
                except queue.Empty:
                    yield (f"⚙️ [{i+1}/{len(pairs)}] 处理中: "
                           f"{os.path.basename(video)}（已用 {time.time()-t0:.0f}s）")
                    continue
                if msg == "__ok__":
                    ok += 1
                    yield f"✅ [{i+1}/{len(pairs)}] 完成: {os.path.basename(out)}"
                    break
                if msg.startswith("__err__"):
                    fail += 1
                    yield (f"⚠️ [{i+1}/{len(pairs)}] 失败: "
                           f"{os.path.basename(video)} -> {msg[7:]}")
                    break
                stage, _, detail = msg.partition(":")
                yield f"⚙️ [{i+1}/{len(pairs)}] {detail or stage}"

        yield f"🏁 烧录结束：成功 {ok}，失败 {fail}。"

    ui_e["burn_button"].click(
        _burn_folder,
        inputs=[
            ui_e["folder_input"],
            ui_e["tts_engine"],
            ui_e["keep_audio"],
            ui_e["sub_font_size"],
            ui_e["sub_font_name"],
            ui_e["sub_position"],
            ui_e["orig_volume"],
            ui_e["tts_volume"],
            ui_e["sub_outline"],
            ui_e["sub_outline_width"],
            ui_e["sub_opacity"],
        ],
        outputs=[burn_status],
    )

    # ---- 参数记忆：保存 / 恢复默认 ----
    # 顺序必须与 SAVE_ORDER 严格一致（否则参数会写串）。
    _settings_widgets = [
        ui_e["folder_input"],
        ui_e["tts_engine"],
        ui_e["keep_audio"],
        ui_e["sub_font_size"],
        ui_e["sub_position"],
        ui_e["sub_font_name"],
        ui_e["sub_outline"],
        ui_e["sub_outline_width"],
        ui_e["sub_opacity"],
        ui_e["orig_volume"],
        ui_e["tts_volume"],
    ]
    assert [w.label for w in _settings_widgets] and len(_settings_widgets) == len(SAVE_ORDER)
    ui_e["save_settings_button"].click(
        save_settings,
        inputs=_settings_widgets,
        outputs=[settings_status],
        queue=False,
    )
    ui_e["reset_settings_button"].click(
        reset_settings,
        inputs=[],
        outputs=_settings_widgets + [settings_status],
        queue=False,
    )

    with gr.Row():
        with gr.Column(scale=1):
            media_output = gr.Image(label="Current Image", interactive=False)
            media_name_markdown = gr.Markdown()
            ui_e["retain_preview_checkbox"] = gr.Checkbox(
                label="Retain preview on skip",
                value=True
            )
        with gr.Column(scale=1):
            caption_output = gr.Textbox(label="Generated Caption", interactive=False)

    ui_e["prompt_input"].change(
        fn=build_final_prompt,
        inputs=[ui_e["prompt_input"], ui_e["summary_mode"], ui_e["one_sentence_mode"]],
        outputs=[prompt_preview],
    )

    ui_e["summary_mode"].change(
        fn=build_final_prompt,
        inputs=[ui_e["prompt_input"], ui_e["summary_mode"], ui_e["one_sentence_mode"]],
        outputs=[prompt_preview],
    )

    ui_e["one_sentence_mode"].change(
        fn=build_final_prompt,
        inputs=[ui_e["prompt_input"], ui_e["summary_mode"], ui_e["one_sentence_mode"]],
        outputs=[prompt_preview],
    )

    ui_e["start_button"].click(
        start_process,
        inputs=[],
        outputs=[ui_e[k] for k in control_keys]
    )

    ui_e["start_button"].click(
        process_folder,
        inputs=[
            ui_e["folder_input"],
            ui_e["prompt_input"],
            ui_e["skip_existing_checkbox"],
            ui_e["max_tokens_slider"],
            ui_e["summary_mode"],
            ui_e["one_sentence_mode"],
            ui_e["retain_preview_checkbox"],
            ui_e["resolution_mode"],
            ui_e["subtitle_mode"],
        ],
        outputs=[
            ui_e["status_output"],
            media_output,
            media_name_markdown,
            caption_output,
            progress_bar,
            time_display,
            *[ui_e[k] for k in control_keys]
        ]
    )

    ui_e["abort_button"].click(
        fn=abort_process,
        inputs=[],
        outputs=[ui_e[k] for k in control_keys],
        queue=False
    )

    ui_e["reset_button"].click(reset_prompt, inputs=[], outputs=[ui_e["prompt_input"]])
    ui_e["start_button"].click(get_model_info, inputs=[], outputs=[model_name_display, device_display, vram_display, dtype_display, config_display])

    ui_e["load_button"].click(
    _ui_load_model,
    inputs=[model_dropdown, custom_model_box, quant_dropdown, attn_dropdown],
    outputs=[ui_e["status_output"], model_name_display, device_display, vram_display, dtype_display, config_display],
    )
    gr.Blocks.load(
        iface,
        get_model_info,
        inputs=[],
        outputs=[model_name_display, device_display, vram_display, dtype_display, config_display]
    )


if __name__ == "__main__":
    # 直接用 python app.py / run_app.bat 启动时才起服务；
    # 被 import 时（如 test_run.py 自检）只构建 UI，不启动服务器。
    #
    # enable_queue / queue 显式打开：process_folder 是生成器，靠 yield 逐步推送进度，
    # 队列关闭时这些中间状态会被合并，界面就看不到实时进度了。
    iface.queue(default_concurrency_limit=1)
    iface.launch(
        share=False,
        theme=gr.themes.Base(),
        css=css,
        inbrowser=True,
    )