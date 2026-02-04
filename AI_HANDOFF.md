# AI 接续说明 (Handoff Notes)

> 给下一个 AI 助手：项目当前状态、本阶段已做事项、历史问题与修复、待办与注意事项。

---

## 1. 项目是做什么的

B站「稍后再看」视频 → 下载音频 → **Whisper 转文字** → **Qwen3（Ollama）校正 + 总结** → 合并成 Markdown → 转 EPUB → 可选上传微信读书。

流程步骤：**A fetch → B download → C transcribe → D correct → E summarize → F epub → G upload**。详见 `PIPELINE_FLOW.md`。

---

## 2. 本阶段我做了哪些事（详细）

### 2.1 Pipeline 改成「按单视频跑完一整条链」

- **之前**：先给所有视频做 B，再集体做 C，再集体做 D…  
- **现在**：对每个视频按顺序做完 B→C→D→E→F（可选 G），再处理下一个。这样单本书会更快出来。
- **实现**：
  - `core/pipeline.py` 里加了 `process_single_video(item, upload=...)`，在一个视频上顺序执行 B/C/D/E/F/G。
  - `run_all(max_items, upload=False)` 先 `run_step_a()` 拉队列，再对前 `max_items` 个视频逐个调用 `process_single_video`。
  - 分步命令（`download` / `transcribe` / `correct` / `summarize` / `epub` / `upload`）保留，可单独重跑某一步。

### 2.2 参数全部走 config，去掉命令行 Whisper 模型

- **之前**：`main.py run --whisper-model large-v3` 等。
- **现在**：Whisper 模型、设备、语言等只在 `config.yaml` 的 `whisper:` 下配置；命令行不再有 `--whisper-model`。
- **代码**：`Pipeline.__init__` 不再收 `whisper_model`；`asr_client` 里用 `config["whisper"]["model"]`（默认 `large-v3`）。

### 2.3 Windows 编码与子进程

- **终端 UTF-8**：`main.py` 里对 Windows 做了 `sys.stdout/stderr.reconfigure(encoding='utf-8')`，避免中文乱码。
- **下载器子进程**：`clients/downloader.py` 里 `subprocess.run(..., encoding="utf-8", errors="replace")`，避免 yt-dlp 输出触发 gbk 解码错误。
- **未做的**：曾给 `subprocess.Popen` 做过 UTF-8 的 monkey-patch，会破坏 asyncio（`windows_utils.Popen` 继承），已**撤销**。若再遇子进程解码错误，只改具体调用处（如 downloader），不要全局改 Popen。

### 2.4 Whisper 模型与缓存路径

- **MemoryError**：用户环境加载 `large-v3` 时曾出现（整模型读入内存做校验）。临时改过 `config` 为 `medium` 以通过；用户后来要求「质量最好」，已改回 **`large-v3`**。
- **No space left on device**：Whisper 默认用 `XDG_CACHE_HOME` 决定下载目录，未用 `WHISPER_CACHE`，导致模型下到 C 盘占满。
- **修改**：在 `main.py` 最前面（在任何 import 前）增加 `os.environ.setdefault("XDG_CACHE_HOME", "E:/ai_models")`，这样 Whisper 的 `download_root` 会是 `E:/ai_models/whisper`，模型和缓存都到 E 盘。
- **当前 config**：`config.yaml` 里 `whisper.model` 为 **`large-v3`**（用户指定要最好质量）。

### 2.5 错误处理与日志

- **Pipeline 异常**：`process_single_video` 里用 `except BaseException`，打印 traceback，并把 `err_msg` + `traceback.format_exc()` 追加到 **`E:\bilibili_summarizer_v3\output\pipeline_error.log`**（若 state 目录不可写则忽略）。
- **状态**：出错时会把该视频的 state 设为 `error` 并写入 `error` 字段，便于排查和重试。

### 2.6 微信读书上传（WeRead）

- **浏览器**：默认用 Playwright 的 **Chromium**（`headless=False` 时可看到窗口）。若要用本机已装的 Chrome，设环境变量 `WEREAD_USE_CHROME=1` 再运行（部分环境用 Chrome 会崩，目前默认 Chromium 更稳）。
- **逻辑**：`clients/weread_browser.py`：启动 persistent context → 打开书架页 → 若未登录则等扫码 → 找「传书」/上传 → 选文件 → 等进度/成功。
- **单独测试上传**：`test_upload.py` 用项目里已有 EPUB 测上传（直接 import weread 模块避免循环 import）。

### 2.7 其它小改动

- **run 命令**：`main.py run` 现在支持 `--upload`，在每视频跑完 F 后执行 G 上传；不再用 `--upload-weread`。
- **PIPELINE_FLOW.md**：已包含 A–G 步骤、输出文件、prompt 路径、config 说明，可与本 handoff 对照。
- **状态模型**：`core/models.py` 里 `VideoState` 已包含 `corrected_md` 等字段，对应 D/E 步产出路径。

---

## 2.8 历史问题与修复记录（供排查参考）

| 问题 | 原因 / 现象 | 已做修复 |
|------|----------------|----------|
| **ImportError: BilibiliClient from clients** | 测试脚本循环 import | 用 `importlib.util.spec_from_file_location` 直接加载模块，避免从 `clients` 包导入 |
| **ModuleNotFoundError: qwen_asr** | 依赖未装 | `pip install qwen-asr` |
| **PyTorch/Transformers 不兼容 + No space left** | C 盘临时目录占满 | 建 venv、指定 torch 轮子、设 `TEMP`/`TMP` 到 `E:\temp` |
| **终端中文乱码** | Windows 默认 gbk | `main.py` 里 `sys.stdout/stderr.reconfigure(encoding='utf-8')`，必要时设 `PYTHONIOENCODING=utf-8` |
| **UnicodeDecodeError: 'gbk' in subprocess** | yt-dlp/ffmpeg 输出被 Python 用 gbk 解码 | `clients/downloader.py` 里 `subprocess.run(..., encoding='utf-8', errors='replace')`；**不要** 对 `subprocess.Popen` 做全局 monkey-patch（会破坏 asyncio 的 `windows_events.Popen`，导致 `TypeError: code must be code, not str`） |
| **Step C 报错但 err_msg 为空** | 异常未完整记录 | `core/pipeline.py` 用 `except BaseException`，打印 traceback，并写入 `pipeline_error.log` |
| **MemoryError 加载 Whisper large-v3** | 模型校验时整文件进内存 | 临时改过 `medium`；用户要求质量最好，已改回 **large-v3**；若再发生可改 config 或建议用户腾内存 |
| **No space left（Whisper 下载）** | Whisper 用默认缓存目录（C 盘） | 在 `main.py` 最前设 `os.environ.setdefault("XDG_CACHE_HOME", "E:/ai_models")`，模型下载到 `E:/ai_models/whisper` |
| **Ollama 卡住/超时** | Step D/E 无统一超时 | 尚未加；建议在 `ollama_client.py` 加 timeout 与重试 |
| **微信读书浏览器看不见 / Chrome 崩** | headless 或 Chrome 兼容性 | 默认用 Playwright Chromium、`headless=False`；用 Chrome 时设 `WEREAD_USE_CHROME=1`（部分环境 Chrome 会崩，文档里已说明） |

---

## 3. 关键文件与配置

| 文件/目录 | 说明 |
|-----------|------|
| `config.yaml` | B站 cookie、下载目录、**whisper**（model/device/language）、**ollama**（model/base_url）、asr_engine、output 路径等。 |
| `main.py` | 入口；最前面设 XDG_CACHE_HOME/WHISPER_CACHE/HF 等；无 Popen 补丁。 |
| `core/pipeline.py` | `run_step_a`，`process_single_video`（B→G 单视频），`run_all`；各 `run_step_*` 仍可单独调用。 |
| `core/state.py` | 状态：new → downloaded → transcribing → transcript_ready → correcting → corrected → summarizing → summarized → success（及 error）。 |
| `clients/whisper_client.py` | Whisper 封装；`transcribe_to_markdown` 含标题/UP主、分段。 |
| `clients/ollama_client.py` | Ollama 调用；`correct_text`（逐段）、`summarize`；`build_final_markdown`。 |
| `clients/downloader.py` | yt-dlp 下载；`subprocess.run` 已带 `encoding="utf-8", errors="replace"`。 |
| `clients/weread_browser.py` | 微信读书上传；`WEREAD_USE_CHROME` 控制是否用系统 Chrome。 |
| `prompts/correct.txt` | 校正 prompt，占位符 `{text}`。 |
| `prompts/summarize.txt` | 总结 prompt，`{title}` / `{author}` / `{text}`。 |
| `E:\bilibili_summarizer_v3\output\` | 产出目录：media/、transcripts/、epub/；state 与 queue：`pipeline_state.json`、`pipeline_queue.json`；错误日志：`pipeline_error.log`。 |

---

## 4. 接下来建议要做 / 可优化的事

### 4.1 必做 / 高优先级

1. **先看当前状态**  
   - 运行：`python main.py status`，确认哪些视频在什么状态（如 `correcting`、`transcribing`、`transcript_ready` 等）。
2. **把「卡在 correcting / transcribing」的视频跑完**  
   - **方案 A**：直接再跑一次：`python main.py run --max-items 1`。若上次停在 `correcting`，会从 D 继续做 D→E→F；若停在 `transcribing`，会从 C 继续。  
   - **方案 B**：若进程已死、Ollama 曾长时间无响应，需要「回退状态」再重跑：  
     - 用 Python 读 `output/pipeline_state.json`，找到对应视频的 `state`，改成可重试状态：  
       - 若已有原始稿 `.md` 文件，可改为 `transcript_ready`，再跑会重新做 D/E/F；  
       - 若已有音频，可改为 `downloaded`，再跑会重新做 C/D/E/F。  
     - 写回 `pipeline_state.json`（务必用 Python/UTF-8，不要用 PowerShell 的 ConvertTo-Json）。  
   - 示例（在项目根目录用 Python 执行）：
     ```python
     import json
     path = "output/pipeline_state.json"
     with open(path, "r", encoding="utf-8") as f:
         data = json.load(f)
     for vid, st in data.get("videos", {}).items():
         if st.get("state") in ("correcting", "transcribing"):
             st["state"] = "transcript_ready"  # 或 "downloaded"
             st.pop("error", None)
     with open(path, "w", encoding="utf-8") as f:
         json.dump(data, f, ensure_ascii=False, indent=2)
     ```
3. **跑通一次完整 E2E（单视频）**  
   - 命令：`python main.py run --max-items 1`（要测上传则加 `--upload`）。  
   - 建议在后台跑或增大超时，避免被工具/终端杀进程（单视频约 15～40 分钟）。  
   - 若 **MemoryError**（加载 large-v3）：临时把 `config.yaml` 里 `whisper.model` 改为 `medium`，或让用户腾内存。  
   - 若 **No space left**：确认 E 盘空间足够，且 `main.py` 里已设 `XDG_CACHE_HOME=E:/ai_models`。  
   - 成功后检查：`output/epub/` 下应有对应 `.epub` 文件；错误可查 `output/pipeline_error.log`。

### 4.2 可选优化

1. **Ollama 超时与重试**  
   Step D/E 若 Ollama 很慢或卡住，当前可能没有统一超时。可在 `clients/ollama_client.py` 里对请求加 timeout，失败时重试 1～2 次，并在 pipeline 里对「校正/总结超时」把状态设为 error 并写入 `pipeline_error.log`。
2. **「correcting / transcribing」卡住后的自动回退**  
   若检测到同一视频长时间处于 `correcting` 或 `transcribing`（例如超过 N 分钟仍无更新），可自动将其改回 `transcript_ready` 或 `downloaded` 并记一条 log，方便下次 `run` 重试。
3. **文档与体验**  
   - 在 README 或 PIPELINE_FLOW 里注明：错误详情看 `output/pipeline_error.log`；微信读书上传默认 Chromium，可选 `WEREAD_USE_CHROME=1` 用本机 Chrome（可能不稳定）。  
   - 可写一句「单视频完整流程大约 15～40 分钟（视长度和机器而定）」，方便用户预期。
4. **Python 版本**  
   若出现 `google.api_core` 等 future 警告，可建议用户升级到 Python 3.11+。

### 4.3 已知限制 / 注意点

- **Whisper large-v3**：需要约 10GB 显存/内存；若机器紧张会 MemoryError，可改用 `medium`。  
- **Ollama**：Step D/E 依赖本地 Ollama 服务（默认 `http://localhost:11434`）；需先启动 Ollama 并拉好 `qwen3:8b`。  
- **长时间运行**：单视频 E2E 可能 20～40 分钟，建议在后台或 nohup 跑，避免终端关闭/超时杀进程。  
- **状态文件**：`pipeline_state.json` 不要用 PowerShell 的 `ConvertTo-Json` 写回，会破坏 UTF-8；用 Python 读写成 JSON。

---

## 5. 常用命令速查

```powershell
# 虚拟环境（若用 E 盘）
E:\bilibili_summarizer_v3\venv\Scripts\activate

# 状态
python main.py status

# 单视频完整 E2E（推荐）
python main.py run --max-items 1

# 单视频 E2E + 上传微信读书
python main.py run --max-items 1 --upload

# 分步重跑（例如只重做校正和之后）
python main.py correct --max-items 1
python main.py summarize --max-items 1
python main.py epub
python main.py upload --max-items 1
```

---

## 6. 小结给下一个 AI

- **流程**：已改成「按单视频跑完 B→F（及可选 G）」；参数从 config 读，无 `--whisper-model`。  
- **环境**：Windows 下终端 UTF-8、下载器子进程 UTF-8；Whisper 缓存用 `XDG_CACHE_HOME` 指到 E 盘；**不要**对 `subprocess.Popen` 做全局补丁（会破坏 asyncio）。  
- **错误**：Pipeline 异常会写 `output/pipeline_error.log` 并更新 state；排查先看该 log 和 `python main.py status`。  
- **接下来**：  
  1. 用 `python main.py status` 看当前状态；  
  2. 若有视频卡在 `correcting`/`transcribing`，可按 §4.1 用 Python 脚本改 state 再重跑，或直接 `run --max-items 1` 看能否续跑；  
  3. 跑通一次单视频 E2E（`run --max-items 1`），必要时临时改用 Whisper `medium` 或腾内存；  
  4. 可选：Ollama 超时/重试、卡住自动回退、文档补充。

---

## 7. 快速上手（下一 AI 第一件事）

1. 读本文件 + `PIPELINE_FLOW.md` + `config.yaml`。  
2. `python main.py status` 看当前 pipeline 状态。  
3. 若有卡住：用 §4.1 的 Python 片段改 `output/pipeline_state.json`，或直接 `python main.py run --max-items 1`。  
4. 跑 E2E：`python main.py run --max-items 1`（长时间，建议后台）；出错看 `output/pipeline_error.log`。

---

*Last updated: 2026-02-02 by AI assistant (handoff for next AI)*
