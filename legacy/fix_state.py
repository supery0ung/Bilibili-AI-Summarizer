import json
from pathlib import Path

path = Path("E:/bilibili_summarizer_v3/output/pipeline_state.json")
if not path.exists():
    print(f"Error: {path} not found")
    exit(1)

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

changed = False
for vid, st in data.get("videos", {}).items():
    if st.get("state") == "summarizing" or st.get("status") == "summarizing":
        print(f"Resetting {vid} from {st.get('status')} to corrected")
        st["status"] = "corrected"
        st.pop("error", None)
        changed = True

if changed:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("Successfully updated pipeline_state.json")
else:
    print("No stuck tasks found to reset")
