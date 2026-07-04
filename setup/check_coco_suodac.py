
# import json
# from pathlib import Path

# json_path = Path(r"data/S-UODAC2020/COCO_Annotations/instances_source.json")

# def main():
#     if not json_path.exists():
#         print(f"[ERROR] 파일 없음: {json_path}")
#         return

#     with open(json_path, "r", encoding="utf-8") as f:
#         d = json.load(f)

#     print("\n=== CATEGORIES ===")
#     print(d.get("categories", "NO CATEGORIES"))

#     print("\n--- IMG ---")
#     if d.get("images"):
#         print(d["images"][0])
#     else:
#         print("NO IMAGES")

#     print("\nDONE")

# if __name__ == "__main__":
#     main()

import json
from pathlib import Path

source_path = Path(r"data/S-UODAC2020/COCO_Annotations/instances_source.json")
target_path = Path(r"data/S-UODAC2020/COCO_Annotations/instances_target.json")

def load_json(path):
    if not path.exists():
        print(f"[ERROR] 파일 없음: {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    s = load_json(source_path)
    t = load_json(target_path)

    if s is None or t is None:
        return

    print("=== IMAGE COUNT ===")
    print("source:", len(s.get("images", [])))
    print("target:", len(t.get("images", [])))

    print("\n=== ANNOTATION COUNT ===")
    print("source:", len(s.get("annotations", [])))
    print("target:", len(t.get("annotations", [])))

    print("\nDONE")

if __name__ == "__main__":
    main()