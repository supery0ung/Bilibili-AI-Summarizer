# Bilibili AI Summarizer

Turn videos from your Bilibili Watch Later list into readable EPUB ebooks.

The pipeline fetches videos, filters them, downloads audio, transcribes speech locally, cleans and summarizes the transcript with a local LLM, builds an EPUB, and can upload the result to WeChat Reading.

## Pipeline

```text
Fetch -> Filter -> Download audio -> Transcribe -> Correct -> Summarize -> EPUB -> Upload
 Step A   Step A       Step B          Step C      Step D     Step E     Step F  Step G
```

| Step | Purpose | Main tools |
| --- | --- | --- |
| A | Fetch Bilibili Watch Later items and filter videos | Bilibili API, rules, Qwen/Ollama |
| B | Download audio | yt-dlp |
| C | Transcribe audio | Qwen3-ASR or Whisper |
| D | Correct transcript text | Ollama |
| E | Generate summary sections | Ollama |
| F | Build EPUB | Pure Python EPUB writer |
| G | Upload EPUB | Playwright browser automation |

## Requirements

- Python 3.10+
- ffmpeg
- yt-dlp
- Ollama with a local model such as `qwen3:8b`
- Optional but recommended: NVIDIA GPU for ASR
- Optional: Playwright browser install for WeChat Reading upload

## Install

```bash
git clone https://github.com/supery0ung/Bilibili-AI-Summarizer.git
cd Bilibili-AI-Summarizer

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

ollama pull qwen3:8b
playwright install chromium
```

On macOS/Linux, activate the virtual environment with:

```bash
source venv/bin/activate
```

## Configure

Copy the example config and fill in your own credentials:

```bash
cp config.example.yaml config.yaml
```

Required Bilibili cookie fields:

- `sessdata`
- `bili_jct`
- `dedeuserid`
- `buvid3`

You can find them in your browser after logging in to Bilibili:

1. Open Bilibili in the browser and log in.
2. Open developer tools.
3. Inspect a request to `api.bilibili.com`.
4. Copy the relevant cookie values into `config.yaml`.

Do not commit `config.yaml`. It contains live credentials and is intentionally ignored by git.

## Model And Cache Paths

By default, the project uses a user cache directory for model and temporary files. You can override that with environment variables:

```bash
set BILIBILI_MODEL_CACHE=D:\ai_models
set BILIBILI_TEMP=D:\temp\bilibili_summarizer
```

On macOS/Linux:

```bash
export BILIBILI_MODEL_CACHE=/data/ai_models
export BILIBILI_TEMP=/data/tmp/bilibili_summarizer
```

Ollama model storage is controlled by Ollama itself.

## Usage

Run the full pipeline:

```bash
python main.py run --max-items 10
```

Run individual steps:

```bash
python main.py fetch
python main.py download --max-items 5
python main.py transcribe --max-items 5
python main.py correct --max-items 5
python main.py summarize --max-items 5
python main.py epub
python main.py upload --max-items 5
```

Process specific videos and bypass the Watch Later queue:

```bash
python main.py run --url https://www.bilibili.com/video/BV1xxxxxx
python main.py run --url BV1xxxxxx BV1yyyyyy
```

Check pipeline status:

```bash
python main.py status
```

## Output

Generated files are written under `output/`:

| Path | Description |
| --- | --- |
| `output/pipeline_state.json` | Resume-safe state for each video |
| `output/pipeline_queue.json` | Current queue |
| `output/media/` | Downloaded audio |
| `output/transcripts/` | Raw, corrected, and final Markdown files |
| `output/epub/` | Generated EPUB ebooks |

The final Markdown file contains:

- Core summary
- Key points
- Full corrected text

Step F converts that final Markdown file into EPUB.

## Tests

The regression suite uses fixtures only. It does not require real Bilibili API calls, downloads, Ollama inference, or GPU work.

```bash
python -m pytest tests/ -v
```

Run a single test file:

```bash
python -m pytest tests/test_epub.py -v
```

## Project Structure

```text
.
├── main.py                  # CLI entry point
├── config.example.yaml      # Safe example config
├── filters.yaml             # Rule-based video filters
├── clients/                 # Bilibili, downloader, ASR, Ollama, WeRead clients
├── core/                    # Pipeline orchestration, state, and step classes
├── prompts/                 # LLM prompt templates
├── scripts/                 # Utility and scheduled-run scripts
├── tests/                   # Regression tests
├── utils/                   # EPUB, logging, state helpers
└── output/                  # Runtime artifacts, ignored by git
```

## Notes

- The pipeline is resume-safe through `output/pipeline_state.json`.
- Prompt behavior can be changed by editing files in `prompts/`.
- `config.yaml`, `output/`, and logs are ignored so private credentials and generated artifacts stay local.
