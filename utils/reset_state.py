import json
from pathlib import Path

def reset_pipeline_state(state_file_path: str):
    state_file = Path(state_file_path)
    if not state_file.exists():
        print(f"Error: State file not found at {state_file_path}")
        return

    print(f"Loading state from {state_file}...")
    data = json.loads(state_file.read_text(encoding="utf-8"))
    videos = data.get("videos", {})

    changes = 0
    for bvid, info in videos.items():
        old_status = info.get("status")
        
        # 1. Handle error states
        if old_status == "error":
            audio_path = info.get("audio_path")
            if audio_path and Path(audio_path).exists():
                info["status"] = "downloaded"
            else:
                info["status"] = "new"
            info.pop("error", None)
            changes += 1
            print(f"  [Reset] {bvid}: error -> {info['status']}")

        # 2. Handle transcribing (rollback to downloaded)
        elif old_status == "transcribing":
            info["status"] = "downloaded"
            changes += 1
            print(f"  [Reset] {bvid}: transcribing -> downloaded")

        # 3. Handle correcting (rollback to transcript_ready)
        elif old_status == "correcting":
            info["status"] = "transcript_ready"
            changes += 1
            print(f"  [Reset] {bvid}: correcting -> transcript_ready")

        # 4. Handle summarizing (rollback to corrected)
        elif old_status == "summarizing":
            info["status"] = "corrected"
            changes += 1
            print(f"  [Reset] {bvid}: summarizing -> corrected")

    if changes > 0:
        state_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Successfully reset {changes} video states.")
    else:
        print("No abnormal states found.")

if __name__ == "__main__":
    # Path from config.yaml
    DEFAULT_STATE_FILE = r"E:\bilibili_summarizer_v3\output\pipeline_state.json"
    reset_pipeline_state(DEFAULT_STATE_FILE)
