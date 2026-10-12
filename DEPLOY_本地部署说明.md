# Simple Captioner 本地部署说明

本机部署记录与使用手册。针对 **Windows + RTX A6000(48GB) + 受限代理网络** 环境。

---

## 一、这是什么

[simple-captioner](https://github.com/o-l-l-i/simple-captioner)（作者 Olli S.，源码可见但非标准开源，禁止再分发/转售/换皮）的本地部署。

用途：给一个文件夹里的**图片/视频批量生成文字描述**，逐张输出同名 `.txt`（`cat.jpg` → `cat.txt`）。
典型场景是给 AI 训练数据打标。

模型：**Qwen3-VL-8B-Instruct**（阿里 Qwen 系列视觉语言模型）。

---

## 二、目录结构

```
simple-captioner/
├── app.py                    # 主程序（Gradio UI + 推理逻辑）
├── requirements.txt          # 原始依赖清单（本机未直接使用，见下方说明）
├── download_model.py         # 从 ModelScope 下载模型
├── download_model.bat        # ↑ 的 Windows 启动器
├── fetch_wheel.py            # 多线程分段下载器（解决单连接限速）
├── setup.bat                 # 一键安装依赖
├── run_app.bat               # 启动程序
├── test_run.py               # 端到端自检（加载模型 + 生成一张图的描述）
├── venv/                     # Python 虚拟环境（Python 3.13）
├── wheels/                   # 手动下载的 torch CUDA 安装包（1.9GB，装完可删）
└── models/
    └── Qwen3-VL-8B-Instruct/ # 模型权重（约 16.3GB）
```

> `wheels/` 里的 wheel 已安装完毕，可安全删除以省 1.9GB 空间；若要重装环境再重新下载即可。

---

## 三、部署验证结果（实测）

| 项目 | 实测值 |
|---|---|
| torch | 2.14.1+cu130（CUDA 13.0） |
| CUDA 可用 | ✅ True，RTX A6000 |
| 可用显存 | 48.0 GB |
| 模型加载耗时 | 17.9 秒 |
| 模型显存占用 | 16.33 GB（FP16） |
| 单张图推理耗时 | 4.4 秒 |
| Gradio 界面 | ✅ `http://127.0.0.1:7860` 返回 200 |

验证方式（可选，随时可跑）：
```
venv\Scripts\python.exe test_run.py
```
会加载模型、画一张测试图、生成英文描述，打通即代表环境正常。

---

## 四、日常使用

首次已由本次部署完成。日常只需：

**双击 `run_app.bat`** → 等模型加载（约 1-2 分钟）→ 浏览器打开 Gradio 界面。

界面操作顺序：
1. **Model Settings** → 点 `📦 Load / Reload Model` 加载模型（首次必须点）
2. 填 **Folder Path**（要处理的图片文件夹，支持子目录）
3. 按需调整 `Custom Prompt` / `Max Tokens` / `Resolution`
4. 点 `🚀 Start Processing`

输出的 `.txt` 会写在**每张图片旁边**。

**量化建议**：48GB 显存可以直接用 `None`（FP16 全精度）。想省显存选 `8-bit`。

---

## 四·补、🎬 字幕解说模式（中文·带时间轴）

界面 **Prompt Controls** 区有个 **`🎬 字幕解说模式（中文·带时间轴）`** 勾选框，**默认勾选**。

**勾选后**：
- **视频** → 输出中文解说字幕，按真实时间轴分段，每行一条，可直接做字幕：
  ```
  [0.0-2.3] 韩国女排奋力救球！
  [2.3-6.0] 高高跃起，力压对手！
  [6.0-9.6] 球飞过网，势如破竹！
  [9.6-13.3] 跑动如风，精准到位！
  [13.3-17.0] 拦网成功，气势如虹！
  [17.0-20.7] 走回场地，淡定从容！
  [20.7-22.0] 比赛未完，精彩继续！
  ```
- **图片** → 输出一句中文画面解说（无时间轴，因为图片没有时间维度）

**取消勾选** → 恢复原版英文描述行为（`Custom Prompt` / Summary / One-Sentence 生效）。

**实现要点**：
- 提示词模板 `SUBTITLE_PROMPT_VIDEO` **会把视频真实时长写进去**（`{duration}`）。
  这一点很关键：若不告诉模型真实时长，它会自己"猜"，时间标记会超出视频长度、无法用于字幕。
- `decode_video_frames()` 因此改为返回三元组 `(frames, sample_fps, duration)`。
- 分段数由 `suggest_segment_count()` 按时长推算（每段约 3 秒，夹在 4~24 段），
  短视频不会拆得太碎，长视频也不会因段数过多被 token 上限截断。
- 字幕模式下 `Max Tokens` 自动抬高到至少 512（多段解说需要更长输出）。
- 字幕模式下 `Custom Prompt` 会被**忽略**（改用专用模板），避免两种指令冲突。

> 输出是**纯字幕文本**，不含序号与英文原文。若要做成标准 SRT，
> 用任意字幕工具（剪映 / Aegisub / Subtitle Edit）粘贴后一键转换时间码即可。

---

## 五、本机环境的三个坑（重要，换 / 重装环境时会再遇到）

### 坑 1：HuggingFace 完全不可达

`huggingface.co` 和 `hf-mirror.com` 在本机网络下都超时。因此：
- 模型改从 **ModelScope（魔搭）** 下载
- `app.py` 增加了 `resolve_model_path()`：加载模型时**优先查 `models/` 下的本地目录**，找不到才回落在线 ID

### 坑 2：pip 所有请求返回 403

**现象**：`pip install` 报 `Could not find a version ... (from versions: none)`，连 gradio 都"找不到"。

**根因**：pip 默认的 User-Agent 是这种格式 ——
```
pip/26.1.2 {"ci":null,"cpu":"AMD64","implementation":{"name":"CPython",...}}
```
其中的 **JSON 花括号和引号被本机 HTTP 代理的 WAF 判定为攻击特征**，一律返回 403。

**修复**：改 `venv/Lib/site-packages/pip/_internal/network/session.py` 第 352 行附近：
```python
# 原：self.headers["User-Agent"] = user_agent()
self.headers["User-Agent"] = "pip/26.1.2"
```

> ⚠️ `pip install --upgrade pip` 升级 pip 后此修改会丢失，需重新打。

### 坑 3：download.pytorch.org 的 403 与限速

**现象 A**：直接下载 wheel 报 403。
**根因**：**URL 里的 `+` 号必须编码成 `%2B`**。裸 `+` 一律 403。
```
✗ https://download.pytorch.org/whl/cu130/torch-2.14.1+cu130-......whl
✓ https://download.pytorch.org/whl/cu130/torch-2.14.1%2Bcu130-......whl
```

**现象 B**：单连接下载只有约 160 KB/s。
**解决**：该源支持 `Accept-Ranges: bytes`，用 `fetch_wheel.py` 开多线程分段下载，实测 6 并发可达 2 MB/s。

### 附：为什么 torch 不从清华源装

清华 PyPI 镜像上的 `torch` 是 **CPU 版**（`2.14.1+cpu`，仅 539MB），装完 `torch.cuda.is_available()` 为 `False`。
CUDA 版只在 `download.pytorch.org`。

本机最终使用：**torch 2.14.1+cu130 + torchvision 0.29.1+cu130**（CUDA 13.0，驱动 597.06 支持）。

---

## 六、常见问题

**Q：`pip install torch torchvision` 会自动装 CPU 版怎么办？**
装其他包（如 accelerate）时 pip 可能顺带装上 CPU 版 torch。要换 CUDA 版必须显式指定：
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```
（单线程很慢，建议用 `fetch_wheel.py` 先下 wheel 再本地安装。）

**Q：模型加载报找不到模型？**
确认 `models/Qwen3-VL-8B-Instruct/config.json` 存在。没有就重跑 `download_model.bat`（ModelScope 支持断点续传）。

**Q：换别的模型？**
```
python download_model.py Qwen/Qwen3-VL-4B-Instruct
```
然后在界面 Model 下拉框选对应项即可（`resolve_model_path` 会自动匹配 `models/` 下的同名目录）。

**Q：视频处理报错？**
原始代码在 Windows 上**视频必然失败**，原因见下方"坑 4"。已修复，无需额外操作。

**Q：点了"🚀 Start Processing"状态栏显示 `Failed to process 1 media (inaccessible, unknown or broken file)`？**
**这不是文件坏了。** 99% 是**模型还没加载完就点了开始处理**。见下方"坑 6"。
正确顺序：先点 **📦 Load / Reload Model** → 等状态栏变成
`✅ Loaded '...' with 8-bit quantization (eager). Ready to process.` → 再点开始处理。

**Q：视频处理时界面卡住很久没反应？**
帧数过高会打满显存导致推理假死，原因见"坑 5"。已把帧数上限压到 24 帧，正常单条视频 10~30 秒完成。

**Q：想提高/降低视频抽帧数或时长？**
改 `app.py` 顶部这几个常量即可：
```python
VIDEO_SAMPLE_FPS = 2.0         # 每秒抽几帧
VIDEO_MAX_FRAMES = 24          # 抽帧上限（⚠️ 调大会显著吃显存，见坑 5）
VIDEO_MAX_DECODE_SECONDS = 60  # 只分析视频前多少秒
VIDEO_MAX_FRAME_SIDE = 640     # 每帧最长边
```

---

## 七、视频相关修复（原版在 Windows 上无法处理视频）

### 坑 4：视频一律失败 —— torchvision 移除了 `read_video`

**现象**：只处理图片，`.mp4` 不生成 `.txt`；控制台报
`AttributeError: module 'torchvision.io' has no attribute 'read_video'`。

**根因链**：
1. `qwen_vl_utils` 读取视频按 `torchcodec → decord → torchvision` 顺序选后端；
2. `decord` **没有 Windows 的 wheel**（PyPI 上仅有 mac/linux），`torchcodec` 未装 → 只能选 `torchvision`；
3. 而 **torchvision 0.19+ 移除了 `io.read_video`**，唯一可用后端一调用就抛异常。

**修复**：不再依赖上述任一后端，改为**用 PyAV 自行解码成帧列表**，再交给
`process_vl_info` 的"帧列表"分支处理（PyAV 的 wheel 自带 FFmpeg 库，装完即可用）。
新增函数 `decode_video_frames()`。

### 坑 5：视频推理"假死" —— 帧数过高打满显存

**现象**：处理视频时界面长时间无响应、无报错、无进度，GPU 占用 100% 但迟迟不结束。

**根因**：8B 模型在 `eager` attention 下，**显存占用随帧数近似线性增长**（实测数据）：

| 帧数 | 分辨率 | 输入 tokens | 峰值显存 |
|---|---|---|---|
| 8 | 960×540 | 2086 | 18.0 GB |
| 16 | 960×540 | 4158 | 21.6 GB |
| 32 | 960×540 | 8308 | **35.2 GB** |
| 53 | 1920×1080 | ~13000 | **超过 48GB → 卡死** |

原始配置会抽到 64 帧且不缩分辨率，48GB 显存被打满后转向内存交换，表现为"卡住"。

**修复**（三道闸门同时收紧）：
1. `VIDEO_MAX_FRAMES` 64 → **24**（24 帧约 26GB，留足余量）；
2. 抽帧后**主动缩放到 `VIDEO_MAX_FRAME_SIDE=640`**，从源头压住每帧 token 数；
3. `VIDEO_MAX_DECODE_SECONDS=60`，只分析视频前 60 秒。

另外修了两处会导致"抽了帧却没被用上"的问题：
- 向 processor 传 `do_sample_frames=False` —— 否则它会按默认 `fps=24` **重新采样，把 32 帧砍成 2 帧**；
- 传 `video_metadata`（`VideoMetadata`）—— 否则模型拿不到真实时间轴，`<x.x seconds>` 时间戳全错。

**实测效果**（RTX A6000 48GB）：

| 素材 | 修复前 | 修复后 | 耗时 | 峰值显存 |
|---|---|---|---|---|
| 26s / 1080p 录屏 | 报错 | ✅ 描述准确 | 10.6 s | 18.4 GB |
| 955s / 4K 录屏 | 报错 | ✅ 描述准确 | 30.3 s | — |

### 视频自检怎么做

```
venv\Scripts\python.exe test_video.py            # 用默认样本
venv\Scripts\python.exe test_video.py D:\a.mp4   # 指定任意视频
```

默认样本固定为 `C:\Users\Starshark_Ci\Videos\2026-09-18 15-20-28.mp4`
（26 秒 1080p 真实录屏，内容为浏览武器资料网站）。

> 之所以不用代码合成的色块视频：合成画面语义太弱，模型会"脑补"成
> 国旗之类的无关内容（实测把蓝底黄块识别成"西班牙国旗"），
> 根本验证不了描述质量。必须用有真实语义的素材。

脚本会打印解码耗时、抽帧数、推理耗时、峰值显存，并把结果写成同名 `.txt`
写到视频旁边（与批处理产物位置一致）。

---

### 坑 6：点了"开始处理"却显示 `Failed to process 1 media (broken file)`

**现象**：选好文件夹、点了"🚀 Start Processing"，状态栏显示
`Failed to process 1 media (inaccessible, unknown or broken file)`，
文件夹里没生成任何 `.txt`。看着像文件损坏，其实**跟文件一点关系都没有**。

**根因**：**模型还没加载完，就点了开始处理。**
- 服务启动后模型**不会自动加载**，需要点 **📦 Load / Reload Model**（8B 模型约 30~60 秒）；
- 用户在加载进度还没走完时点"开始处理"，此时 `model is None`，
  `generate_caption()` 直接抛 `AssertionError: Model must be loaded before generating captions.`；
- 这个异常被 `process_folder` 的 `except` 捕获，只打印到控制台，
  UI 上则被汇总成那句**极具误导性**的 "inaccessible, unknown or broken file"。

用 HTTP API 直连复现时，服务端日志看得一清二楚（真实报错）：
```
[DEBUG] folder_path:  E:\Desktop\测试内容
[ERROR] Failed processing file E:\Desktop\测试内容\xxx.mp4:
        Model must be loaded before generating captions.
```

**修复**（`app.py`）：
1. 新增全局标志 `model_loading`，`load_selected_model()` 用 `try/finally` 维护它；
2. `process_folder()` 开头做**前置检查**，分两种提示，不再逐文件误报：
   - 从未加载 → `⚠️ No model loaded. Please click "📦 Load / Reload Model" first...`
   - 正在加载 → `⏳ Model is still loading, please wait... (takes ~40s)`
3. 异常信息改为**透传真实错误**（`类型: 具体信息`），最终汇总也带上 `First error ->`；
4. `_ui_load_model` 改成生成器，点击后**立即**显示 `⏳ Loading ... please wait`，
   不会让用户以为卡死而反复点击；
5. 页面顶部加了一行醒目提示，明确"先加载模型、等状态栏变 Ready 再处理"。

**实测验证**（RTX A6000 / 8-bit）：

| 操作时机 | 修复前 | 修复后 |
|---|---|---|
| 冷启动、未点加载就处理 | ❌ `Failed to process 1 media (broken file)` | ✅ `⚠️ No model loaded...`（0.1s 返回） |
| 模型加载中途点处理 | ❌ 同样的误导信息 | ✅ `⏳ Model is still loading, please wait...` |
| 模型就绪后点处理 | ✅ 正常 | ✅ 正常（29s，`.txt` 正确落盘） |

> 排查提示：`Failed to process ... (broken file)` 这句话是原版的**兜底文案**，
> 任何异常都会走到这里。**先看服务端控制台的 `[ERROR]` 行**，那里才是真实原因。

---

## 八、本次部署对源码的改动

| 文件 | 改动 |
|---|---|
| `app.py` | 新增 `MODELS_ROOT` 常量与 `resolve_model_path()`，`load_selected_model()` 改为优先加载本地模型目录 |
| `app.py` | 新增 `decode_video_frames()`（PyAV 解码，见坑 4），`generate_caption()` 改走帧列表分支 |
| `app.py` | 视频参数收紧 + 传 `do_sample_frames=False`/`video_metadata`/`cap_pixels_per_frame`（见坑 5） |
| `app.py` | 新增 `model_loading` 标志 + `process_folder()` 模型就绪前置检查 + 错误信息透传（见坑 6） |
| `app.py` | `_ui_load_model` 改为生成器（加载中即时反馈）；页面顶部新增操作顺序提示（见坑 6） |
| `app.py` | 新增字幕解说模式：`SUBTITLE_PROMPT_VIDEO/IMAGE`、`suggest_segment_count()`、`generate_caption(..., subtitle_mode=)`、`process_folder(..., subtitle_mode=)`、UI 勾选框 `🎬 字幕解说模式`（见"四·补"） |
| `app.py` | `decode_video_frames()` 返回值增加 `duration`（字幕需真实时长） |
| `app.py` | `launch()` 加 `inbrowser=True`，启动后自动打开浏览器 |
| `venv/.../pip/_internal/network/session.py` | 替换 User-Agent（见坑 2） |
| 新增 `download_model.py` / `.bat` | ModelScope 模型下载 |
| 新增 `fetch_wheel.py` | 多线程分段下载器 |
| 新增 `test_run.py` | 图片端到端自检脚本 |
| 新增 `test_video.py` | 视频端到端自检脚本 |
| 新增 `test_video.py` | 视频端到端自检脚本（默认样本为 26s 真实录屏） |
| 新增 `setup.bat` / 重写 `run_app.bat` | 一键安装与启动（原 `run_app.bat` 从 GitHub 下载时被 CDN 拦截，内容为空） |
