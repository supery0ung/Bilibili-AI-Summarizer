# Bilibili Summarizer V3 - Whisper 转录版

将 B站"稀后再看"视频自动转录为文字，生成 EPUB 电子书，上传微信读书。

## 功能

1. **获取稀后再看列表** - 通过 Bilibili API
2. **过滤视频** - 按时长、UP主、标题规则过滤
3. **下载视频音频** - 使用 yt-dlp 下载
4. **语音转文字** - 使用本地 Whisper 模型转录
5. **生成 EPUB** - 转换为电子书格式
6. **上传微信读书** - 自动上传 (可选)

## 安装

### 1. 安装 Python 依赖

```bash
cd bilibili_summarizer_v3
pip install -r requirements.txt
```

### 2. 安装 ffmpeg (Whisper 需要)

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
```bash
# 使用 chocolatey
choco install ffmpeg
# 或下载: https://ffmpeg.org/download.html
```

### 3. 安装 yt-dlp (视频下载)

```bash
pip install yt-dlp
# 或
brew install yt-dlp
```

## 配置

1. 复制配置示例:
   ```bash
   cp config.example.yaml config.yaml
   ```

2. 填写配置:
   - Bilibili cookies (从浏览器获取)
   - Whisper 模型选择
   - 其他可选配置

### Whisper 模型选择

| 模型 | VRAM | 速度 | 质量 | 适用场景 |
|------|------|------|------|----------|
| `tiny` | ~1 GB | 最快 | 较低 | 测试 |
| `base` | ~1 GB | 快 | 一般 | 简单内容 |
| `small` | ~2 GB | 中 | 良好 | 日常使用 |
| `medium` | ~5 GB | 慢 | 很好 | **推荐** |
| `large-v3` | ~10 GB | 最慢 | 最佳 | 高端 GPU |

在 `config.yaml` 中设置:
```yaml
whisper:
  model: "medium"  # 推荐
  device: "auto"   # 自动检测 GPU/CPU
```

## 使用

```bash
# 运行完整流水线
python main.py run

# 只获取 + 过滤视频列表
python main.py fetch

# 只下载 + 转录 (最多处理 10 个)
python main.py transcribe --max-items 10

# 使用指定 Whisper 模型
python main.py transcribe --whisper-model large-v3

# 只生成 EPUB
python main.py epub

# 上传到微信读书
python main.py upload

# 查看当前状态
python main.py status
```

## 获取 Bilibili Cookies

1. 打开浏览器，登录 bilibili.com
2. 按 F12 打开开发者工具
3. 切换到 Network 标签
4. 刷新页面，找到任意 api.bilibili.com 请求
5. 在 Request Headers 中找到 Cookie，提取:
   - `SESSDATA`
   - `bili_jct`
   - `DedeUserID`
   - `BUVID3`

## 文件结构

```
bilibili_summarizer_v3/
├── main.py              # CLI 入口
├── config.yaml          # 配置文件 (需自行创建)
├── filters.yaml         # 过滤规则
├── clients/             # 客户端模块
│   ├── bilibili.py      # B站 API
│   ├── downloader.py    # 视频下载 (yt-dlp)
│   ├── whisper_client.py # Whisper 转录
│   └── weread_browser.py # 微信读书上传
├── core/                # 核心逻辑
│   ├── pipeline.py      # 流水线编排
│   ├── state.py         # 状态管理
│   └── models.py        # 数据模型
├── utils/               # 工具函数
└── output/              # 输出目录
    ├── media/           # 下载的音频文件
    ├── summaries/       # 转录文本 (.md)
    └── epub/            # 生成的电子书
```

## GPU 加速

### macOS (Apple Silicon)
Whisper 会自动使用 MPS 加速。

### Windows (NVIDIA GPU)
确保安装了 CUDA 版本的 PyTorch:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```
