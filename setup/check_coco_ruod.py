import json
from pathlib import Path

json_path = Path(r"data/RUOD/RUOD_ANN/instances_train.json")

def main():
    if not json_path.exists():
        print(f"[ERROR] 파일 없음: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        d = json.load(f)

    print("\n=== CATEGORIES ===")
    print(d.get("categories", "NO CATEGORIES"))

    print("\nDONE")

if __name__ == "__main__":
    main()