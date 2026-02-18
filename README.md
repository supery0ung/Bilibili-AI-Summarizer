# Bilibili Summarizer V3

Automatically transcribe Bilibili "Watch Later" videos to text, perform AI correction and summarization using **Qwen ASR**, generate EPUB ebooks, and upload to WeChat Reading.

将 B站"稍后再看"视频自动转录为文字，使用 **Qwen ASR** 进行 AI 校正与总结，生成 EPUB 电子书，上传微信读书。

## Workflow / 功能流程

```
Fetch List → Filter → Download Audio → Qwen ASR Transcription → LLM Correction → LLM Summary → Generate EPUB → Upload to WeChat Reading
获取列表 → 过滤 → 下载音频 → Qwen ASR 转录 → LLM 校正 → LLM 总结 → 生成 EPUB → 上传微信读书
```

| Step | Function / 功能 | Technology / 技术 |
|------|------|------|
| Step A | Fetch & Filter / 获取 + 过滤 | Bilibili API + Qwen3 |
| Step B | Download Audio / 下载音频 | yt-dlp (Parallel) |
| Step C | ASR Transcription / 语音转文字 | **Qwen3-ASR** / Whisper (Local) |
| Step D | AI Correction / 文档校正 | Qwen3 8B (Local Ollama) |
| Step E | AI Summary / 内容总结 | Qwen3 8B (Local Ollama) |
| Step F | Generate EPUB / 生成电子书 | Pure Python |
| Step G | Upload to WeRead / 上传微信读书 | Playwright Automation |

## Requirements / 环境要求

- Python 3.10+
- NVIDIA GPU (12GB+ VRAM Recommended)
- [Ollama](https://ollama.com/) + `qwen3:8b` model
- ffmpeg
- yt-dlp

## 安装

```bash
cd bilibili_summarizer_v3
pip install -r requirements.txt

# 安装 ASR 和 LLM 模型
ollama pull qwen3:8b
```

## 配置

```bash
cp config.example.yaml config.yaml
```

编辑 `config.yaml`，填写：
- Bilibili cookies（SESSDATA, bili_jct 等）
- ASR 引擎选择（`qwen_asr` 或 `whisper`）
- Ollama 模型配置

## 使用

```powershell
# 一键运行全部流程 (A → G)
python main.py run --max-items 10

# 分步运行
python main.py fetch                    # Step A: 获取 + 过滤
python main.py download --max-items 5   # Step B: 下载音频
python main.py transcribe --max-items 5 # Step C: 语音转文字
python main.py correct --max-items 5    # Step D: 校正文本
python main.py summarize --max-items 5  # Step E: 生成摘要
python main.py epub                     # Step F: 生成 EPUB
python main.py upload --max-items 5     # Step G: 上传微信读书

# 查看状态
python main.py status
```

## 输出文件

| 步骤 | 文件 | 说明 |
|------|------|------|
| Step C | `{标题}.md` | ASR 原始转录 |
| Step D | `{标题}.corrected.md` | 校正后文本 |
| Step E | `{标题}.final.md` | 摘要 + 校正文本 |
| Step F | `{标题}.epub` | 电子书 |

## 项目结构

```
bilibili_summarizer_v3/
├── main.py                     # CLI 入口
├── config.yaml                 # 配置文件（需自行创建）
├── config.example.yaml         # 配置示例
├── filters.yaml                # 视频过滤规则
├── requirements.txt            # Python 依赖
├── PIPELINE_FLOW.md            # 流程详细说明
├── TESTING.md                  # 测试文档
│
├── clients/                    # 外部服务客户端
│   ├── bilibili.py             # Bilibili API
│   ├── downloader.py           # yt-dlp 音频下载
│   ├── ollama_client.py        # Qwen3 LLM（校正+总结）
│   ├── qwen_asr_client.py      # Qwen3-ASR 语音识别
│   └── weread_browser.py       # 微信读书上传
│
├── core/                       # 核心流水线逻辑
│   ├── pipeline.py             # 流水线编排
│   ├── state.py                # 状态管理（pipeline_state.json）
│   ├── models.py               # 数据模型（VideoInfo, VideoState, QueueItem）
│   ├── filter.py               # 视频过滤器
│   ├── base_step.py            # 步骤基类
│   ├── step_downloader.py      # Step B: 下载
│   ├── step_asr.py             # Step C: 转录
│   └── step_llm.py             # Step D+E: 校正 + 总结
│
├── utils/                      # 工具函数
│   ├── md_to_epub.py           # Markdown → EPUB 转换
│   ├── logger.py               # 日志配置
│   └── reset_state.py          # 状态重置工具
│
├── prompts/                    # LLM Prompt 模板
│   ├── correct.txt             # 文本校正
│   ├── summarize.txt           # 内容总结
│   ├── filter.txt              # AI 智能过滤
│   └── identify_speakers.txt   # 说话人识别
│
├── tests/                      # 回归测试（41 个测试）
│   ├── conftest.py             # 共享 fixture
│   ├── test_models.py          # 数据模型测试
│   ├── test_state.py           # 状态管理测试
│   ├── test_filter.py          # 过滤器测试
│   ├── test_step_llm.py        # LLM 步骤测试
│   ├── test_epub.py            # EPUB 生成测试
│   ├── test_md_to_epub.py      # Markdown 转换测试
│   └── test_build_final_md.py  # 最终文档结构测试
│
└── output/                     # 输出目录
    ├── pipeline_state.json     # 流水线状态
    ├── pipeline_queue.json     # 当前队列
    ├── media/                  # 下载的音频
    ├── transcripts/            # 转录文本 (.md)
    └── epub/                   # 生成的电子书
```

## 获取 Bilibili Cookies

1. 打开浏览器，登录 bilibili.com
2. 按 F12 打开开发者工具 → Network 标签
3. 刷新页面，找到 `api.bilibili.com` 请求
4. 在 Request Headers → Cookie 中提取：`SESSDATA`、`bili_jct`、`DedeUserID`、`BUVID3`

## 模型配置

| 模型 | 用途 | 配置位置 |
|------|------|----------|
| Qwen3-ASR | 语音识别 | `config.yaml` → `asr_engine: qwen_asr` |
| Qwen3 8B | 文本校正 + 总结 | Ollama: `qwen3:8b` |

模型存储位置：
- HuggingFace: `E:/ai_models/huggingface/`
- Ollama: `E:/ai_models/ollama/`

## 测试

```powershell
& "e:\bilibili_summarizer_v3\venv\Scripts\python.exe" -m pytest tests/ -v
```

详细测试说明见 [TESTING.md](TESTING.md)。
