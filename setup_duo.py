"""
setup_duo.py — DUO 를 ultralytics 학습용으로 준비
=================================================
자동 감지:
  - YOLO txt 라벨이 이미 있으면 → duo.yaml 만 생성
  - COCO json 만 있으면        → COCO→YOLO 변환 후 duo.yaml 생성

DUO 구조 (organize 후):
  data/DUO/images/train2017/*.jpg
  data/DUO/images/test2017/*.jpg
  data/DUO/annotations/instances_train2017.json
  data/DUO/annotations/instances_test2017.json

ultralytics 가 기대하는 최종 구조:
  data/DUO/images/train2017/1.jpg
  data/DUO/labels/train2017/1.txt      ← YOLO txt (class cx cy w h, 정규화)
  data/DUO/images/test2017/...
  data/DUO/labels/test2017/...
  + duo.yaml

사용:
  python setup_duo.py --root data/DUO
"""

import argparse
import json
import os
import glob
from collections import defaultdict


def find_coco_json(ann_dir, split):
    """split(train/test)에 해당하는 COCO json 찾기."""
    if not os.path.isdir(ann_dir):
        return None
    for f in glob.glob(os.path.join(ann_dir, "*.json")):
        low = os.path.basename(f).lower()
        if split in low:
            return f
    return None


def coco_to_yolo(coco_json, img_dir, label_dir):
    """COCO json → YOLO txt 라벨 생성. 반환: (변환 이미지 수, category 정보)."""
    with open(coco_json, encoding="utf-8") as f:
        data = json.load(f)

    # category id → 0-based 연속 인덱스
    cats = sorted(data.get("categories", []), key=lambda c: c["id"])
    cat_map = {c["id"]: i for i, c in enumerate(cats)}
    cat_names = [c["name"] for c in cats]

    # 이미지 정보
    imgs = {im["id"]: im for im in data["images"]}

    # 이미지별 annotation 모으기
    anns_by_img = defaultdict(list)
    for a in data.get("annotations", []):
        anns_by_img[a["image_id"]].append(a)

    os.makedirs(label_dir, exist_ok=True)
    n_written = 0
    for img_id, im in imgs.items():
        W, H = im["width"], im["height"]
        stem = os.path.splitext(os.path.basename(im["file_name"]))[0]
        lines = []
        for a in anns_by_img.get(img_id, []):
            x, y, w, h = a["bbox"]                       # COCO: top-left + wh (절대)
            if w <= 0 or h <= 0:
                continue
            cx = (x + w / 2) / W
            cy = (y + h / 2) / H
            nw = w / W
            nh = h / H
            # 클램프 (경계 넘는 박스 방지)
            cx, cy = min(max(cx, 0), 1), min(max(cy, 0), 1)
            nw, nh = min(nw, 1), min(nh, 1)
            cls = cat_map[a["category_id"]]
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        # 라벨 파일 작성 (객체 없어도 빈 파일 — ultralytics 는 negative 로 인식)
        with open(os.path.join(label_dir, f"{stem}.txt"), "w") as lf:
            lf.write("\n".join(lines))
        n_written += 1
    return n_written, cat_names


def has_yolo_labels(label_dir):
    """label_dir 에 .txt 가 이미 있는지."""
    if not os.path.isdir(label_dir):
        return False
    return len(glob.glob(os.path.join(label_dir, "*.txt"))) > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/DUO", help="DUO 루트")
    ap.add_argument("--yaml-out", default="duo.yaml")
    ap.add_argument("--force-convert", action="store_true",
                    help="YOLO txt 있어도 COCO 에서 재변환")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    ann_dir = os.path.join(root, "annotations")

    # ultralytics 폴더 규칙: images/<split> ↔ labels/<split>
    splits = {"train": "train2017", "test": "test2017"}
    cat_names = None

    for split, folder in splits.items():
        img_dir = os.path.join(root, "images", folder)
        label_dir = os.path.join(root, "labels", folder)

        if not os.path.isdir(img_dir):
            print(f"[warn] 이미지 폴더 없음: {img_dir} (스킵)")
            continue

        n_imgs = len(glob.glob(os.path.join(img_dir, "*.jpg")))

        if has_yolo_labels(label_dir) and not args.force_convert:
            n_txt = len(glob.glob(os.path.join(label_dir, "*.txt")))
            print(f"[{split}] YOLO txt 이미 존재: {n_txt} labels "
                  f"(images {n_imgs}) → 변환 스킵")
            # category 이름은 json 에서 한 번 읽어둠
            if cat_names is None:
                cj = find_coco_json(ann_dir, split)
                if cj:
                    with open(cj, encoding="utf-8") as f:
                        d = json.load(f)
                    cats = sorted(d.get("categories", []), key=lambda c: c["id"])
                    cat_names = [c["name"] for c in cats]
        else:
            cj = find_coco_json(ann_dir, split)
            if cj is None:
                print(f"[{split}] COCO json 도 없고 YOLO txt 도 없음 → 스킵")
                continue
            print(f"[{split}] COCO→YOLO 변환: {os.path.basename(cj)}")
            n, names = coco_to_yolo(cj, img_dir, label_dir)
            cat_names = names
            print(f"[{split}] {n} labels 생성 (images {n_imgs})")

    if cat_names is None:
        print("\n[ERROR] category 정보를 못 얻음. COCO json 확인 필요.")
        return

    # duo.yaml 작성
    # ultralytics 는 path 기준 상대경로. train/val 을 images 폴더로 지정.
    yaml_text = f"""# DUO dataset for ultralytics (auto-generated by setup_duo.py)
path: {root.replace(os.sep, '/')}
train: images/train2017
val: images/test2017
test: images/test2017

nc: {len(cat_names)}
names:
"""
    for i, name in enumerate(cat_names):
        yaml_text += f"  {i}: {name}\n"

    with open(args.yaml_out, "w", encoding="utf-8") as f:
        f.write(yaml_text)

    print(f"\n생성 완료: {os.path.abspath(args.yaml_out)}")
    print(f"  nc={len(cat_names)}, names={cat_names}")
    print(f"\n다음:")
    print(f"  python train_ppcm_yolo.py --data {args.yaml_out} "
          f"--weights yolov8m.pt --stage1 0 --stage2 0 --dataset-name DUO")


if __name__ == "__main__":
    main()