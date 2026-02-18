# 测试文档

## 概述

本项目使用 **pytest** 进行回归测试，共 **44 个测试**，覆盖流水线的每个关键环节。所有测试使用 fixture 数据，无需 GPU、Ollama 或网络连接，运行时间 < 1 秒。

## 运行方式

```bash
cd d:\Dropbox\Coding\Bilibili_Summarizer\bilibili_summarizer_v3
& "e:\bilibili_summarizer_v3\venv\Scripts\python.exe" -m pytest tests/ -v
```

## 测试策略

### 核心原则

| 原则 | 说明 |
|---|---|
| **Fixture 驱动** | 所有测试使用预构建的样本数据，不调用真实 API/LLM |
| **隔离性** | 每个测试使用 `tmp_path`（临时目录），无副作用 |
| **聚焦契约** | 每个测试验证一个具体的输入→输出契约 |

### 不测什么

- ❌ 不测试 Bilibili API 调用（需要登录 cookie）
- ❌ 不测试 Ollama/LLM 推理（需要 GPU）
- ❌ 不测试音频下载和 ASR 转录（需要网络和模型）

### 测什么

- ✅ 数据在各步骤间的 **格式和流转**
- ✅ 状态管理的 **持久化和转换**
- ✅ EPUB **最终输出内容**（摘要是否存在）
- ✅ 过滤规则的 **正确性**

## 测试模块详解

### 1. `test_models.py` — 数据模型（4 个测试）

验证 `VideoState`、`QueueItem`、`VideoInfo` 的序列化/反序列化。

```
test_roundtrip        → to_dict() → from_dict() 保持所有字段不丢失
test_defaults         → 默认值正确（status="new", summary_md=None）
test_from_api_response → 能正确解析 Bilibili API 返回的 dict
```

**为什么重要**: 如果序列化丢失了 `summary_md` 字段，EPUB 中就不会包含摘要。

### 2. `test_state.py` — 状态管理（6 个测试）

验证 `StateManager` 的持久化、查询、队列构建。

```
test_update_and_get      → 写入后能正确读取
test_summary_md_persists → summary_md 路径在保存/加载后保持不变
test_get_pending_items   → 按状态筛选 bvid 列表
test_full_status_flow    → new → downloaded → ... → uploaded 完整流转
test_skips_uploaded      → build_queue 跳过已上传的视频
test_includes_error      → build_queue 包含出错的视频（用于重试）
```

**为什么重要**: 状态是流水线的"记忆"，状态丢失或错误会导致重复处理或跳过视频。

### 3. `test_filter.py` — 视频过滤（4 个测试）

验证 `VideoFilter` 的各种过滤规则。

```
test_filter_short_duration → 短于 min_seconds 的视频被过滤
test_filter_up_deny        → 黑名单 UP 主被过滤
test_filter_title_regex    → 标题匹配正则时被过滤
test_filter_keeps_valid    → 正常视频通过所有过滤
```

### 4. `test_step_llm.py` — LLM 步骤契约（5 个测试）

**不调用真实 LLM**，而是模拟 `_correct_item` 和 `_summarize_item` 的文件 I/O 和状态更新契约。

```
test_produces_corrected_md      → 生成 .corrected.md 文件，状态变为 "corrected"
test_corrected_file_format      → 文件包含 # 标题、**UP主**、--- 分隔
test_saves_summary_md_to_state  → 生成 .final.md 并写入 summary_md 状态字段
test_final_md_has_summary_sections → 包含 "核心摘要"、"要点列表"、"完整文本"
test_status_flow                → transcript_ready → corrected → summarized 流转
```

**为什么重要**: 之前的 bug 就是因为 `summary_md` 没写入状态，导致 EPUB 拿不到摘要。

### 5. `test_epub.py` — EPUB 生成（7 个测试）⚠️ 最关键

```
test_epub_contains_summary_sections → EPUB 中包含 "核心摘要"、"要点列表"、"完整文本"
test_epub_contains_body_text        → EPUB 中包含正文内容
test_epub_from_transcript_still_works → 只有原始转录（无摘要）时也能生成
test_prioritizes_summary_md         → 同时存在时，优先使用 summary_md
test_falls_back_to_transcript_md    → summary_md 为空时，回退到 transcript_md
test_valid_zip                      → EPUB 是合法 ZIP，包含所有必需文件
test_title_in_metadata              → content.opf 中有正确的 <dc:title>
test_xhtml_well_formed              → content.xhtml 有 XML 声明和完整标签
```

### 6. `test_md_to_epub.py` — Markdown 转换（11 个测试）

验证 `md_to_html()` 和 `safe_filename()` 工具函数。

```
# md_to_html
test_headings, test_bold_italic, test_lists, test_blockquote,
test_horizontal_rule, test_inline_code, test_paragraph

# safe_filename
test_special_chars_removed, test_length_capped,
test_chinese_preserved, test_empty_returns_untitled
```

### 7. `test_build_final_md.py` — 最终 Markdown 结构（3 个测试）

验证 `build_final_markdown()` 输出格式。

```
test_structure          → 包含 # 标题、**UP主**、---、摘要、完整文本
test_summary_before_body → 摘要在 "完整文本" 之前
test_chinese_content     → 中文内容正确保留
```

### 8. `test_bilingual.py` — 双语校对逻辑（3 个测试）

验证对于非中文转录的识别和双语输出处理。

```
test_bilingual_hint_triggered_for_english → 识别到英文时，提示词中包含双语要求
test_no_bilingual_hint_for_chinese       → 识别到中文时，不强制双语
test_language_heuristic_in_qwen_client    → 验证 Qwen3ASRClient 的启发式语言识别准确度
```

## 测试数据

所有测试共享一组中文 fixture 数据（定义在 `conftest.py`）：

- **标题**: `测试视频：AI 技术分析`
- **作者**: `测试UP主`
- **正文**: 4 段关于人工智能的样本文本
- **摘要**: 包含 `核心摘要`、`要点列表`、`总结与建议` 三个标准章节

这些 fixture 通过 `tmp_path` 写入临时文件，测试结束后自动清理。
