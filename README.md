# Simple Captioner（本地部署增强版）

> 基于 [o-l-l-i/simple-captioner](https://github.com/o-l-l-i/simple-captioner) 的 Windows 本地部署版本。
> 原版在 Windows 上**无法处理视频**，本版做了修复，并新增**中文解说字幕模式**。

用 Qwen3-VL 视觉语言模型给**图片和视频批量生成文字说明**，
每处理一个文件，就在它旁边写出一个**同名字幕/文本文件**。

- 🖼️ 图片 → 一句中文描述，写 `.txt`
- 🎬 视频 → **标准 `.srt` 字幕**，按真实时间轴分段，播放器/剪辑软件直接加载
- 📦 批量处理整个文件夹（含子目录）
- 📊 **实时进度反馈**，处理过程中随时可见完成了多少

---

## 目录

1. [环境要求](#一环境要求)
2. [快速开始（三步）](#二快速开始三步)
3. [要下载的模型](#三要下载的模型)
4. [界面使用详解](#四界面使用详解)
5. [字幕解说模式](#五字幕解说模式)
6. [合成旁白视频（烧录字幕+TTS）](#五补合成旁白视频烧录字幕--tts-解说)
7. [输出说明](#六输出说明)
8. [自检脚本](#七自检脚本)
9. [常见问题](#八常见问题)
10. [给 AI 的部署提示词](#九给-ai-的部署提示词)
11. [项目结构](#十项目结构)

---

## 一、环境要求

| 项目 | 要求 | 本机实测 |
|---|---|---|
| 操作系统 | Windows 10 / 11 | Windows |
| Python | 3.10 ~ 3.13 | 3.13.12 |
| 显卡 | **NVIDIA，显存 ≥ 12GB** | RTX A6000 48GB |
| 磁盘 | ≥ 30GB 可用空间 | 模型 17GB + 环境 4GB |
| 网络 | 能访问 PyPI / ModelScope | 需代理绕过 GitHub、HuggingFace |

**显存参考**（8B 模型，8-bit 量化）：

| 场景 | 峰值显存 |
|---|---|
| 图片 | 约 10 GB |
| 短视频（24 帧 640px） | 约 12.5 GB |
| 长视频（24 帧 640px） | 约 13 GB |

> ⚠️ **没有 NVIDIA 显卡也能跑，但会非常慢**（CPU 推理）。此时建议改用 4B 模型。

---

## 二、快速开始（三步）

### 第 1 步：安装依赖

双击 **`setup.bat`**

它会自动：
1. 创建虚拟环境 `venv/`
2. 安装 PyTorch（CUDA 版）
3. 安装其余依赖（gradio / transformers / qwen_vl_utils / modelscope 等）

> 耗时取决于网速，PyTorch CUDA 包约 2GB。

### 第 2 步：下载模型

双击 **`download_model.bat`**

- 从 **ModelScope（魔搭）** 下载 `Qwen/Qwen3-VL-8B-Instruct`，约 **17GB**
- 支持**断点续传**，中断后重跑即可继续
- 下载到 `models/Qwen3-VL-8B-Instruct/`

### 第 3 步：启动

双击 **`run_app.bat`**

浏览器会自动打开 `http://127.0.0.1:7860`。

**然后必须按顺序操作：**

```
① 点 【📦 Load / Reload Model】  →  等待 30~60 秒
② 等状态栏出现：✅ Loaded '...' ... Ready to process.
③ 填 Folder Path
④ 点 【🚀 Start Processing】
```

> ❗ **最容易踩的坑**：模型没加载完就点"开始处理"，会提示
> `No model loaded` 或 `Model is still loading`。
> 这不是文件坏了，**先加载模型再处理**即可。

---

## 三、要下载的模型

### 默认模型（推荐）

| 项目 | 值 |
|---|---|
| **模型 ID** | `Qwen/Qwen3-VL-8B-Instruct` |
| **下载源** | **ModelScope（魔搭）** |
| **体积** | 约 17 GB |
| **本地路径** | `models/Qwen3-VL-8B-Instruct/` |
| **下载方式** | 双击 `download_model.bat` |

> **为什么用 ModelScope 而不是 HuggingFace？**
> 本机网络无法访问 `huggingface.co` 和 `hf-mirror.com`（连接超时）。
> ModelScope 是国内源，速度可达 20MB/s。
> 如果你的网络能访问 HF，也可以直接改 `download_model.py` 用 `huggingface-cli`。

### 手动下载命令

```bash
# 默认 8B 模型
venv\Scripts\python.exe download_model.py

# 换其它模型
venv\Scripts\python.exe download_model.py Qwen/Qwen3-VL-4B-Instruct
```

### 可选模型

界面 Model 下拉框内置以下选项（都需要先下载到 `models/` 下同名目录）：

| 模型 | 体积 | 说明 |
|---|---|---|
| `Qwen/Qwen3-VL-4B-Instruct` | ~9 GB | 显存紧张时首选，12GB 卡可跑 |
| `Qwen/Qwen3-VL-8B-Instruct` | ~17 GB | **默认，效果最好** |
| `Qwen/Qwen2.5-VL-7B-Instruct` | ~16 GB | 上一代，兼容性好 |
| `Qwen/Qwen2.5-VL-3B-Instruct` | ~7 GB | 最轻量 |

> 模型目录名**必须**与 ID 最后一段一致
> （`Qwen/Qwen3-VL-8B-Instruct` → `models/Qwen3-VL-8B-Instruct/`），
> `app.py` 的 `resolve_model_path()` 按这个约定查找。
> 用 `download_model.py` 下载会自动符合约定。

### 模型加载优先级

```
优先：models/<模型名>/    ← 本地目录（有 config.json 就用它）
回落：在线 ID 直接拉取     ← 本机网络下会失败
```

---

## 四、界面使用详解

### 操作顺序

| 步骤 | 位置 | 操作 |
|---|---|---|
| 1 | **Model Settings** | 选模型 → 选量化 → 点 `📦 Load / Reload Model` |
| 2 | | **等状态栏显示 `✅ ... Ready to process.`** |
| 3 | **Folder Path** | 填要处理的文件夹，如 `D:\我的图片` |
| 4 | **Prompt Controls** | 按需勾选（见下） |
| 5 | | 点 `🚀 Start Processing` |
| 6 | **状态栏 / 进度条** | 观察 `📊 总进度 (%)` 实时推进 |

### 进度反馈（实时可见）

点下 `🚀 Start Processing` 后，界面会**持续刷新**，不会"卡住没反应"：

- **📊 总进度 (%)**：按文件数推进，单个文件内部还会按阶段细分：
  `解码抽帧 10% → 构建输入 20% → 模型推理 85% → 收尾 100%`
- **⏱️ 总耗时 (mm:ss)**：从点下按钮开始计时
- **上方状态栏**：显示当前是第几个文件、当前处于哪个阶段，例如
  - `⏳ [1/3] 开始处理: xxx.mp4`
  - `⚙️ [1/3] 正在解码视频并抽取关键帧 ...`
  - `⚙️ [1/3] 已抽取 24 帧（时长 22.0s）`
  - `⚙️ [1/3] 正在运行模型推理 ... 已用时 12s`（推理期间**每秒心跳**刷新）
  - `✅ 处理完成！共 3 个文件；成功 3，跳过 0，失败 0，总耗时 01:43。`

实测参考（RTX A6000 48GB，8-bit，3 段共约 57s 的短视频）：首个进度事件 **0.1s** 内出现，
整批 123 次进度刷新，总耗时 1 分 43 秒。

### 关键参数

| 参数 | 可选值 | 建议 |
|---|---|---|
| **Model** | 见上文可选模型 | `Qwen3-VL-8B-Instruct` |
| **Quantization** | `None` / `8-bit` / `4-bit` | **`8-bit`**（省显存，质量几乎无损）；48GB 卡可选 `None` |
| **Attention** | `flash_attention_2` / `eager` | Windows 用 **`eager`**（flash-attn 在 Windows 编译困难） |
| **Skip already captioned** | 勾选/不勾 | 已有 `.srt`/`.txt` 的跳过，**重跑时建议勾上** |
| **Max Tokens** | 32 ~ 1024 | 默认 512；字幕模式会自动抬到 ≥512 |
| **Image Resolution** | `auto` / `auto_high` / `fast` / `high` | 默认 `auto` 即可 |
| **Retain preview on skip** | 勾选/不勾 | 跳过文件时保留上一张预览 |

### Prompt Controls 三个勾选框

| 选项 | 作用 |
|---|---|
| **Summary Mode** | 让模型简要概括（英文） |
| **One-Sentence Mode** | 限制为一句短句（英文） |
| **🎬 字幕解说模式** | **中文解说 + 时间轴**，详见下一节（**默认勾选**） |

> 勾了「字幕解说模式」后，`Summary Mode` / `One-Sentence Mode` / `Custom Prompt` 会被**忽略**，
> 因为改用专用中文模板。取消勾选即恢复英文描述模式。

---

## 五、字幕解说模式

**默认开启**。这是把输出直接变成可用的字幕文件。

### 视频输出 → 标准 `.srt`

视频会生成同名 **`.srt`** 字幕文件，格式完全符合 SRT 规范，
播放器（PotPlayer / VLC）和剪辑软件（剪映 / Premiere / DaVinci）都能**直接加载**：

```
1
00:00:00,000 --> 00:00:03,200
跳台之上，她蓄势待发。

2
00:00:03,200 --> 00:00:07,100
一个完美的倒立，优雅至极。

3
00:00:07,100 --> 00:00:10,500
身体舒展，准备起跳。

4
00:00:10,500 --> 00:00:14,300
她调整呼吸，专注凝神。

5
00:00:14,300 --> 00:00:15,900
一切准备就绪，即将入水。
```

例：`D:\media\a.mp4` → `D:\media\a.srt`

### 图片输出 → `.txt`

图片没有时间维度，所以输出**一句中文画面解说**（不超过 25 字），写同名 `.txt`。

### 技术要点（改代码时会用到）

| 机制 | 说明 |
|---|---|
| **真实时长注入** | 视频真实时长会写进提示词（`{duration}`）。**不给时长模型会自己瞎猜**，时间轴就对不上 |
| **分段数自适应** | `suggest_segment_count()` 按「时长 ÷ 3 秒」计算，夹在 **4~24 段**。短视频不拆太碎，长视频不超 token |
| **Token 自动抬高** | 字幕模式下 `Max Tokens` 自动抬到 ≥ 512，防止多段解说被截断 |
| **时长来源** | `decode_video_frames()` 返回三元组 `(frames, sample_fps, duration)` |
| **SRT 由代码转换** | 模型只输出好写的 `[起-止] 文本`，由 `caption_to_srt()` 精确格式化成 `00:00:00,000`。**时间码绝不语法出错** |
| **解析容错** | `_SUBTITLE_LINE_RE` 兼容：无方括号、全角【】、全角波浪号 `～`、中文破折号 `—`、行首多余序号、混入说明文字 |
| **自动修正** | 时间倒序会自动排序；起点重叠会自动顺延；零长度区间会自动补足 1 秒 |
| **失败回退** | 若一行时间轴都没解析出来（模型没按格式输出），**自动回退写 `.txt`**，不会产生空文件 |

**为什么不让模型直接输出 SRT？**
SRT 时间码语法严格（`HH:MM:SS,mmm`，必须逗号、必须补零），让模型直接写容易出错；
而且模型算毫秒级时间本来就不准。让模型专注「看懂画面 + 写好解说词」，
时间码交给程序计算，是更稳的分工。

---

## 五·补、🎥 合成旁白视频（烧录字幕 + TTS 解说）

跑完字幕处理后，界面上有个 **`🎥 开始烧录字幕+旁白`** 按钮，可以把 `.srt`
直接做成**带硬字幕和中文语音解说的成片**（`_旁白版.mp4`），发出去就能看。

### 它做什么

1. 遍历文件夹，找到所有「视频 + 同名 `.srt`」组合
2. 把每条解说词用 TTS 合成语音，按字幕起始时间铺到音轨上
3. ffmpeg 把字幕烧录进画面（样式按下方设置）
4. 旁白与原声混音（音量按下方设置）

### Web 端可调参数

所有设置支持**记忆功能**：调好后点「💾 保存当前设置」写入项目根目录的
`settings.json`（已加入 .gitignore），下次启动自动回填；「↩️ 恢复默认值」
只回填界面，点保存后才会落盘。

| 设置项 | 范围 | 默认 | 说明 |
|---|---|---|---|
| 🔠 字幕大小 | 8 ~ 96 | 20 | ASS 字号；1080p 竖屏建议 20~32，横屏建议 28~44 |
| 📍 字幕位置 | 底部居中/靠左/靠右、画面正中、顶部居中 | 底部居中 | 旧版 ffmpeg（<4.4）按 SSA 对齐表自动换算 |
| 🔤 字体 | 常见中文字体 + 可手输 | Microsoft YaHei | 需系统已安装该字体 |
| 🖊️ 字幕描边 | 开/关 + 粗细 0~4 | 开 / 2 | 关闭后为纯色文字，无黑边 |
| 🫥 字幕不透明度 | 0 ~ 1.0 | 1.0 | 0.6 左右可让字幕透出画面 |
| 🎬 视频原声音量 | 0 ~ 2.0 | 0.5 | 0 = 静音原声 |
| 🗣️ TTS 解说音量 | 0 ~ 3.0 | 1.0 | 解说太小声就调大 |
| 保留视频原声 | 开/关 | 开 | 关闭则成片只有解说音轨 |

### 语音引擎

| 引擎 | 音质 | 联网 | 说明 |
|---|---|---|---|
| **edge（默认）** | ⭐⭐⭐⭐⭐ 接近真人 | 需要 | 微软在线语音（晓晓等 8 种中文音色）；**被代理/防火墙拦截时自动回退 sapi** |
| **sapi** | ⭐⭐ 机械音 | 不需要 | Windows 自带的 Huihui 中文女声，完全离线 |

### ffmpeg 说明

需要带 **libass**（`subtitles` 滤镜）和 **libx264** 的 ffmpeg。
程序按以下顺序探测：

1. 环境变量 `SC_FFMPEG` 指定的路径
2. 系统 PATH 里的 `ffmpeg`
3. `D:\EVCapture\ffmpeg.exe`（EVCapture 自带，实测可用）
4. `D:\JianyingPro\...\ffmpeg.exe`（剪映自带，部分版本是阉割版）

没有 ffmpeg？去 https://www.gyan.dev/ffmpeg/builds/ 下载 `ffmpeg-release-full`，
解压后把 `bin` 目录加进 PATH，或设置 `SC_FFMPEG` 环境变量。

### 实测数据

| 视频 | 旁白条数 | 耗时 | 输出 |
|---|---|---|---|
| 沙滩排球 22.0s（文件名带引号） | 7 | 11s | ✅ |
| 跳台跳水 15.9s | 5 | 7s | ✅ |
| video_1080p 18.9s | 6 | 9s | ✅ |

### 实现要点（改代码时）

- 逻辑全部在 **`burner.py`**，`app.py` 只是调用（延迟导入，不影响启动速度）
- **字幕路径转义坑**：文件名含单引号会破坏 ffmpeg 滤镜的引号语法
  → 先把 `.srt` 复制成临时安全文件名再引用
- **老版本 ffmpeg 兼容**：`amix` 的 `normalize` 参数在 4.4 之前不存在，带上直接报错
  → 用 `volume` 滤镜手动控制音量
- **对齐值按版本换算**：旧版 ffmpeg（<4.4，如 git-2019）按 SSA 对齐表解释
  `Alignment`（实测 6=顶部居中、10=画面正中），新版按 ASS 数字键盘（8=顶、5=中）
  → `_position_params()` 内部按 `ffmpeg_version()` 自动选择
- **透明度坑**：ASS 颜色 `&HAABBGGRR` 里的 AA 是「透明度」且方向相反
  （00=完全不透明，FF=全透明）→ 代码里用 `(1 - 不透明度) * 255` 反算
- 关描边时要把 `Outline` 和 `Shadow` 都设 0，只设 Outline=0 仍会残留阴影黑边
- **参数记忆的顺序坑**：`save_settings`/`reset_settings` 按界面控件顺序收值，
  键名映射靠 `SAVE_ORDER` 列表对应。**调整界面控件顺序时必须同步改
  `SAVE_ORDER` 和 `_settings_widgets`**，否则参数会写串（曾实测踩坑）
- **Gradio 生成器必须走队列**：`queue=False` 的事件**不支持生成器** ——
  函数里写 `yield` 的话，界面只会收到第一条消息就永远停住，后面的代码
  根本不执行（表现为点「保存当前设置」一直卡在"正在保存"）。
  → 保存这类毫秒级同步操作直接用普通函数 `return` 最终结果
- SAPI 输出 mono/22050Hz，edge-tts 输出 24000Hz
  → 混音前用 `aresample=44100` + `aformat` 统一

---

## 六、输出说明

| 场景 | 输出文件 | 内容 |
|---|---|---|
| 视频 + 字幕模式 | 同名 **`.srt`** | 标准 SRT 字幕（序号 + 时间码 + 解说词） |
| 图片（任意模式） | 同名 **`.txt`** | 一句中文解说 |
| 视频 + 关闭字幕模式 | 同名 **`.txt`** | 英文描述（原版行为） |
| 视频 + 烧录旁白按钮 | 同名 **`_旁白版.mp4`** | 硬字幕 + TTS 解说混音成片 |
| 视频 + 字幕模式但解析失败 | 同名 **`.txt`**（回退） | 模型原始输出，便于排查 |

- 编码统一 UTF-8
- **支持子目录**，会递归处理
- **勾了 `Skip already captioned`** 时，视频会检查 `.srt` 是否存在，图片检查 `.txt`
- 支持的格式：
  - 图片：`png` `jpg` `jpeg` `bmp` `gif` `webp`
  - 视频：`mp4` `mov` `avi` `webm` `mkv` `gif` `flv`

---

## 七、自检脚本

装完后想确认环境正常，可以跑自检：

```bash
# 图片自检（端到端：加载模型 → 生成 → 写 .txt）
venv\Scripts\python.exe test_run.py

# 视频自检（打印解码耗时/抽帧数/推理耗时/峰值显存）
venv\Scripts\python.exe test_video.py
venv\Scripts\python.exe test_video.py D:\your_video.mp4   # 指定视频
```

---

## 八、常见问题

### Q：点了"开始处理"显示 `Failed to process 1 media (broken file)`？

**这不是文件坏了。** 99% 是**模型还没加载完就点了开始处理**。

> 这句是原版的**兜底文案**，任何异常都会显示成它。

正确顺序：先点 `📦 Load / Reload Model` → 等状态栏变 `✅ ... Ready to process.` → 再处理。

### Q：视频处理时卡住很久没反应？

**先看界面上的进度**：`📊 总进度 (%)` 和状态栏里的 `正在运行模型推理 ... 已用时 Ns` 会每秒刷新。
如果这两个数字在动，说明程序正常，只是模型推理慢（8-bit 8B 模型处理 24 帧约 25~40 秒/条）。

如果进度**完全不动**且超过 2 分钟，才是真的卡住 —— 通常是显存被抽帧数打满了。
本项目已把帧数上限压到 **24 帧**并缩放到 **640px**，正常单条视频 10~40 秒完成。

想调整，改 `app.py` 顶部常量：

```python
VIDEO_SAMPLE_FPS = 2.0         # 每秒抽几帧
VIDEO_MIN_FRAMES = 4           # 最少抽帧数
VIDEO_MAX_FRAMES = 24          # 最多抽帧数（⚠️ 调大会显著吃显存）
VIDEO_MAX_DECODE_SECONDS = 60  # 只分析视频前多少秒
VIDEO_MAX_FRAME_SIDE = 640     # 每帧最长边
```

### Q：`pip install` 报 403 / `from versions: none`？

某些企业代理会把 pip 默认 User-Agent 里的花括号当成可疑请求拦掉。
解决：编辑 `venv\Lib\site-packages\pip\_internal\network\session.py`，
把 `self.headers["User-Agent"]` 设成一个普通字符串，如 `"pip/26.1.2"`。

> 注意：`pip install --upgrade pip` 会覆盖掉这个补丁，需要重新打。

### Q：`download.pytorch.org` 报 403？

URL 里的 `+` 必须编码成 `%2B`（例如 `cu130` → `2.14.1%2Bcu130`）。
本项目的 `fetch_wheel.py` 已处理此问题，可多线程分段下载（单线程约 160KB/s，多线程可达 2MB/s）。

### Q：清华源装的 torch 是 CPU 版？

清华 PyPI 的 torch 是 **CPU-only**（约 539MB，`cuda.is_available()` 返回 False）。
**CUDA 版只能从 `download.pytorch.org` 装。**

### Q：模型加载报找不到模型？

确认 `models/Qwen3-VL-8B-Instruct/config.json` 存在。没有就重跑 `download_model.bat`。

### Q：想把描述改回英文？

取消勾选 **🎬 字幕解说模式**，然后在 `Custom Prompt` 里填想要的英文提示词。

### Q：端口 7860 被占用？

```bash
# 查看占用
netstat -ano | findstr :7860
# 结束进程（换成实际 PID）
taskkill /F /PID <PID>
```

---

## 九、给 AI 的部署提示词

> **下次换机器时，把下面这段直接发给 AI 助手，它就能照着部署。**

```text
请帮我在这台 Windows 机器上部署 simple-captioner（本地版）。

【目标】
用 Qwen3-VL 给图片和视频批量生成中文解说字幕。
视频输出同名 .srt 标准字幕文件，图片输出同名 .txt 描述。

【环境】
- Python 3.10~3.13，NVIDIA 显卡（显存 ≥12GB）
- 网络受限：GitHub、HuggingFace 不可达；PyPI、ModelScope 可达

【步骤】
1. 创建虚拟环境：python -m venv venv
2. 装 CUDA 版 PyTorch（关键：必须从 download.pytorch.org 装，清华源是 CPU 版）：
   venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
3. 装其余依赖：
   venv\Scripts\python.exe -m pip install "gradio>=6.10.0" "Pillow>=11.2.1" "qwen_vl_utils>=0.0.11" "transformers>=5.4.0" "accelerate>=0.26.0" "bitsandbytes>=0.46.1" "hf_xet>=1.1.10" modelscope
4. 从 ModelScope 下载模型（约 17GB，支持断点续传）：
   venv\Scripts\python.exe download_model.py Qwen/Qwen3-VL-8B-Instruct
   下载到 models/Qwen3-VL-8B-Instruct/
5. 启动：venv\Scripts\python.exe app.py
   浏览器打开 http://127.0.0.1:7860

【已知坑（务必注意）】
- torchvision 0.19+ 移除了 io.read_video，qwen_vl_utils 读视频会全线失败
  → 本项目已改用 PyAV 自行解码（decode_video_frames），需确保 av 包已安装
- 视频抽帧必须传 do_sample_frames=False，否则 processor 会按 fps=24 重新采样，
  把 24 帧砍到 2 帧
- 必须传 video_metadata（VideoMetadata），否则模型拿到的时间轴是错的
- 显存随帧数近似线性增长：24 帧约 26GB，53 帧 1080p 会超过 48GB 卡死
- pip 默认 User-Agent 可能被代理拦截返回 403，需改成普通字符串
- 模型加载需要 30~60 秒，加载完成前点"开始处理"会报 No model loaded
- 进度反馈依赖 iface.queue(default_concurrency_limit=1)，
  少了这句 generator 的 yield 不会流到前端，界面会一直转圈

【验证】
venv\Scripts\python.exe test_run.py     # 图片自检
venv\Scripts\python.exe test_video.py   # 视频自检

如果上面任何一步失败，请先看控制台的 [ERROR] 行，那里是真实原因。
```

---

## 十、项目结构

```
simple-captioner/
├── app.py                      # 主程序（Gradio 界面 + 推理逻辑）
├── burner.py                   # 旁白视频合成（SRT 解析 + TTS + ffmpeg 烧录混音）
├── requirements.txt            # 依赖清单
├── setup.bat                   # 一键装依赖
├── download_model.bat          # 一键下模型（ModelScope）
├── download_model.py           # 模型下载脚本
├── run_app.bat                 # 一键启动（Windows）
├── run_app.sh                  # 启动脚本（Linux/macOS）
├── fetch_wheel.py              # 多线程 wheel 下载器（网络差时用）
├── test_run.py                 # 图片端到端自检
├── test_video.py               # 视频端到端自检
├── DEPLOY_本地部署说明.md       # 详细部署手册（含全部踩坑记录）
├── .gitignore                  # 排除模型/环境/wheel 等大文件
│
├── models/                     # ⛔ 不上传：模型权重 17GB（download_model.bat 重建）
├── venv/                       # ⛔ 不上传：虚拟环境 4.3GB（setup.bat 重建）
└── wheels/                     # ⛔ 不上传：预下载 wheel 1.9GB（fetch_wheel.py 重建）
```

**上传 GitHub 时只需源码，约 54 KB。**
模型、虚拟环境、wheel 全部由脚本重建，已在 `.gitignore` 中排除。

---

## 与上游原版的差异

| 改动 | 原因 |
|---|---|
| **PyAV 解码视频** | 原版依赖 torchvision `read_video`，该函数在 0.19+ 已被移除 → Windows 上视频必然失败 |
| **视频抽帧参数收紧** | 原版抽 64 帧不缩分辨率，48GB 显存被打满导致假死 |
| **传 `do_sample_frames=False`** | 否则 processor 会重新采样，把抽好的帧丢掉 94% |
| **传 `VideoMetadata`** | 否则模型拿不到真实时间轴，时间戳全错 |
| **`resolve_model_path()`** | 优先加载本地 `models/` 目录，适配 HF 不可达的环境 |
| **🎬 字幕解说模式** | 新增功能：中文解说 + 真实时间轴。视频导出**标准 `.srt`** 字幕文件 |
| **`caption_to_srt()`** | 模型只输出 `[起-止] 文本`，由代码精确转成 SRT 时间码；解析失败自动回退 `.txt` |
| **🎥 旁白视频合成** | 新增 `burner.py`：把 `.srt` 烧录进画面 + TTS 旁白混入音轨，输出 `_旁白版.mp4` |
| **📊 实时进度反馈** | 新增功能：进度条 + 阶段状态 + 每秒心跳，处理短视频也能看到进度 |
| **模型加载前置检查** | 原版模型未加载时误报"文件损坏"，现改为明确提示 |
| **加载中即时反馈** | `Load Model` 改为生成器，点击后立刻显示 `⏳ Loading ...` |

---

## 许可

沿用原仓库许可，见 `LICENSE.txt`。
原项目作者：[Olli S.](https://github.com/o-l-l-i)
