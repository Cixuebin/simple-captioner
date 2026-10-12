# Simple Captioner（本地部署增强版）

> 基于 [o-l-l-i/simple-captioner](https://github.com/o-l-l-i/simple-captioner) 的 Windows 本地部署版本。
> 原版在 Windows 上**无法处理视频**，本版做了修复，并新增**中文解说字幕模式**。

用 Qwen3-VL 视觉语言模型给**图片和视频批量生成文字说明**，
每处理一个文件，就在它旁边写出一个**同名 `.txt`**。

- 🖼️ 图片 → 一句中文/英文描述
- 🎬 视频 → **中文解说字幕**，按真实时间轴分段，可直接做字幕
- 📦 批量处理整个文件夹（含子目录）

---

## 目录

1. [环境要求](#一环境要求)
2. [快速开始（三步）](#二快速开始三步)
3. [要下载的模型](#三要下载的模型)
4. [界面使用详解](#四界面使用详解)
5. [字幕解说模式](#五字幕解说模式)
6. [输出说明](#六输出说明)
7. [自检脚本](#七自检脚本)
8. [常见问题](#八常见问题)
9. [给 AI 的部署提示词](#九给-ai-的部署提示词)
10. [项目结构](#十项目结构)

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

### 关键参数

| 参数 | 可选值 | 建议 |
|---|---|---|
| **Model** | 见上文可选模型 | `Qwen3-VL-8B-Instruct` |
| **Quantization** | `None` / `8-bit` / `4-bit` | **`8-bit`**（省显存，质量几乎无损）；48GB 卡可选 `None` |
| **Attention** | `flash_attention_2` / `eager` | Windows 用 **`eager`**（flash-attn 在 Windows 编译困难） |
| **Skip already captioned** | 勾选/不勾 | 已有 `.txt` 的跳过，**重跑时建议勾上** |
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

### 视频输出

按**真实时间轴**分段，每行一条，格式 `[起-止] 解说词`（时间为秒，保留一位小数）：

```
[0.0-2.3] 韩国女排奋力救球！
[2.3-6.0] 高高跃起，力压对手！
[6.0-9.6] 球飞过网，势如破竹！
[9.6-13.3] 跑动如风，精准到位！
[13.3-17.0] 拦网成功，气势如虹！
[17.0-20.7] 走回场地，淡定从容！
[20.7-22.0] 比赛未完，精彩继续！
```

### 图片输出

图片没有时间维度，所以只输出**一句中文画面解说**（不超过 25 字）。

### 技术要点（改代码时会用到）

| 机制 | 说明 |
|---|---|
| **真实时长注入** | 视频真实时长会写进提示词（`{duration}`）。**不给时长模型会自己瞎猜**，时间轴就对不上 |
| **分段数自适应** | `suggest_segment_count()` 按「时长 ÷ 3 秒」计算，夹在 **4~24 段**。短视频不拆太碎，长视频不超 token |
| **Token 自动抬高** | 字幕模式下 `Max Tokens` 自动抬到 ≥ 512，防止多段解说被截断 |
| **时长来源** | `decode_video_frames()` 返回三元组 `(frames, sample_fps, duration)` |

### 做成标准 SRT

输出是**纯文本**。要做成 SRT，任选其一：

- **剪映**：导入视频 → 文本 → 智能字幕 → 粘贴文本 → 使用"文稿匹配"自动打轴
- **Subtitle Edit / Aegisub**：粘贴后一键转换时间码格式
- 需要的话可以让 AI 写个小脚本把 `[起-止]` 批量转成 SRT

---

## 六、输出说明

- 每个文件旁生成 **同名 `.txt`**，编码 UTF-8
- 例：`D:\media\a.mp4` → `D:\media\a.txt`
- **支持子目录**，会递归处理
- 支持的格式：
  - 图片：`png` `jpg` `jpeg` `bmp` `gif` `webp`
  - 视频：`mp4` `mov` `avi` `webm` `mkv` `gif` `flv`

---

## 七、自检脚本

装完后想确认环境正常，可以跑自检：

```bash
# 图片自检（端到端：加载模型 → 生成 → 写 txt）
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

显存被抽帧数打满了。本项目已把帧数上限压到 **24 帧**并缩放到 **640px**，正常单条视频 10~40 秒完成。

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
用 Qwen3-VL 给图片和视频批量生成中文解说字幕，输出同名 .txt 到媒体文件旁边。

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
| **🎬 字幕解说模式** | 新增功能：中文解说 + 真实时间轴，可直接做字幕 |
| **模型加载前置检查** | 原版模型未加载时误报"文件损坏"，现改为明确提示 |
| **加载中即时反馈** | `Load Model` 改为生成器，点击后立刻显示 `⏳ Loading ...` |

---

## 许可

沿用原仓库许可，见 `LICENSE.txt`。
原项目作者：[Olli S.](https://github.com/o-l-l-i)
