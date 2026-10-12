import os
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


IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')
VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi', '.webm', '.mkv', ".gif", ".flv")
DEFAULT_PROMPT = "Describe this media."
DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_QUANT = "8-bit"  # "None" | "8-bit" | "4-bit"
DEFAULT_ATTN = "eager" if os.name == "nt" else "flash_attention_2"

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
                     resolution_mode="auto", subtitle_mode=False):
    global processor, model
    assert model is not None, "Model must be loaded before generating captions."
    assert processor is not None, "Processor must be loaded before generating captions."

    ext = os.path.splitext(media_path)[-1].lower()
    is_video = ext in VIDEO_EXTENSIONS

    # ---- 解码媒体：视频需要先解码，因为解说模式要用真实时长构建提示词 ----
    video_meta = None
    video_duration = None
    if is_video:
        # 用 PyAV 解码成帧列表（规避 torchvision 0.29 已移除 read_video 的坑）
        media_data, sample_fps, video_duration = decode_video_frames(media_path)
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


    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)

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
    if model is None or processor is None:
        control_updates = finish_process()
        if model_loading:
            msg = ("⏳ Model is still loading, please wait... "
                   "(the model auto-loads on startup and takes ~40s; "
                   "do NOT click Start Processing until the status bar shows the model is ready)")
        else:
            msg = ("⚠️ No model loaded. Please click \"📦 Load / Reload Model\" first, "
                   "then wait until loading finishes before clicking Start Processing.")
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
            txt_filename = os.path.splitext(os.path.basename(media_path))[0] + ".txt"
            txt_path = os.path.join(os.path.dirname(media_path), txt_filename)

            rel_path = os.path.relpath(media_path, folder_path)
            media_name_markdown = f"**File:** `{rel_path}`"

            if skip_existing and os.path.exists(txt_path):
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

            # 解说字幕模式下，多段解说需要更长输出，自动抬高上限（至少 512 tokens）
            eff_max_tokens = max(max_tokens, 512) if subtitle_mode else max_tokens
            caption = generate_caption(media_path, prompt, eff_max_tokens, summary_mode,
                                       one_sentence_mode, resolution_mode, subtitle_mode)

            # Only generate previews for images
            if is_image_file(media_path):
                media_to_show = Image.open(media_path)
            else:
                media_to_show = None

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
            yield (
                f"🖼️ Processing {idx+1}/{total_media}: {rel_path}",
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
    total_time = f"{end_time - start_time:.2f}"


    control_updates = finish_process()
    if failed_media > 0:
        # 有失败时，把真实错误原因带出来，而不是笼统地说"文件损坏"
        summary = (f"⚠️ Done. processed {processed_media}, skipped {skipped_media}, "
                   f"failed {failed_media}. First error -> {first_error}")
    else:
        summary = (f"✅ Processing complete! , processed {processed_media} media in {elapsed_str}, "
                   f"skipped {skipped_media} media. Failed to process {failed_media} media")
    yield (
        summary,
        last_media_to_show,
        last_media_name_markdown,
        last_caption,
        None,
        None,
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

    with gr.Row():
        ui_e["folder_input"] = gr.Textbox(label="📁 Folder Path", placeholder="e.g. C:\\Users\\you\\Pictures\\input_images")
        ui_e["prompt_input"] = gr.Textbox(label="Custom Prompt", value=DEFAULT_PROMPT)

    ui_e["skip_existing_checkbox"] = gr.Checkbox(label="Skip already captioned media (.txt exists)", value=False)

    with gr.Row():
        gr.Markdown("### Prompt Controls")
        gr.Markdown("""
        - **Summary Mode**: Asks the model to summarize the media content briefly.
        - **One-Sentence Mode**: Instructs the model to keep the caption to a single concise sentence.
        - **字幕解说模式 (Subtitle)**: 输出中文解说字幕；视频按真实时间轴分段，格式 `[起-止] 解说词`，可直接做字幕。
        """)
        ui_e["summary_mode"] = gr.Checkbox(label="Summary Mode", value=False)
        ui_e["one_sentence_mode"] = gr.Checkbox(label="One-Sentence Mode", value=False)
        ui_e["subtitle_mode"] = gr.Checkbox(label="🎬 字幕解说模式（中文·带时间轴）", value=True)

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
    progress_bar = gr.Slider(minimum=0, maximum=100, label="Progress", interactive=False)
    time_display = gr.Textbox(label="⏱️ Time Taken (s)", interactive=False)

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
    iface.launch(share=False, theme=gr.themes.Base(), css=css, inbrowser=True)