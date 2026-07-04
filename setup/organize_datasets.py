"""
organize_datasets.py  (개선판)
================================
RUOD / DUO / S-UODAC2020 의 제각각인 압축 해제 구조를 검증2·학습용 표준 구조로 정리.
Brackish 는 Roboflow YOLO 버전을 별도로 받으므로 이 스크립트 대상에서 제외
(원하면 --only 로도 안 부르면 됨).

개선점 (이전 버전 대비):
  1. 파일명 충돌 감지: 같은 basename 이 두 번 나오면 조용히 skip 되던 것을 경고로 표시
  2. organize 후 최종 카운트 자동 검산 (기대치와 비교)
  3. 완료 후 검증2용 datasets.yaml 자동 생성 (경로 불일치 에러 원천 차단)
  4. unrouted(어느 split 에도 못 간 이미지) 상세 표시

사용:
  python organize_datasets.py --dry-run              # 미리보기
  python organize_datasets.py                        # 복사 실행
  python organize_datasets.py --move                 # 복사 대신 이동(원본 삭제)
  python organize_datasets.py --only RUOD DUO        # 일부만
  python organize_datasets.py --make-yaml            # organize 후 datasets.yaml 생성
"""

import argparse
import json
import shutil
from pathlib import Path
from collections import defaultdict

# ── 각 데이터셋의 "압축 푼 원본" 최상위 폴더 ──────────────────────────
SOURCES = {
    "RUOD":         r"./data/RUOD",
    "DUO":          r"./data/DUO",
    "S-UODAC2020":  r"./data/S-UODAC2020",
    # "Brackish":   Roboflow YOLO 버전을 data/Brackish 에 직접 배치 (organize 불필요)
}

DATA_ROOT = Path("./data")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
VID_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
JUNK_PARTS = {"__MACOSX"}

# organize 후 기대 이미지 수 (검산용; 대략치, 버전따라 ±수십 가능)
EXPECTED = {
    "RUOD": 14000,
    "DUO": 7782,
    "S-UODAC2020": 5542,
}

MOVE = False
DRY = False

# 파일명 충돌 추적: dst 경로별로 몇 번 place 시도됐는지
_seen_dst = defaultdict(int)
_collisions = []


def is_junk(p: Path) -> bool:
    if p.name.startswith("._"):
        return True
    return any(part in JUNK_PARTS for part in p.parts)


def place(src: Path, dst: Path) -> str:
    # 충돌 감지: 같은 목적지에 서로 다른 원본이 오면 경고
    _seen_dst[str(dst)] += 1
    if _seen_dst[str(dst)] > 1:
        _collisions.append((str(src), str(dst)))
        return "collision"
    if dst.exists() and not DRY:
        return "skip"
    if DRY:
        return "plan"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if MOVE:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(src, dst)
    return "move" if MOVE else "copy"


def walk_files(root: Path, exts=None, exclude=()):
    for p in root.rglob("*"):
        if not p.is_file() or is_junk(p):
            continue
        if exts is not None and p.suffix.lower() not in exts:
            continue
        low = str(p).lower()
        if any(e in low for e in exclude):
            continue
        yield p


def split_of(text: str):
    t = text.lower()
    if "train" in t:
        return "train"
    if "test" in t:
        return "test"
    if "val" in t:
        return "val"
    return None


def coco_basenames(json_path: Path):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"    [warn] COCO json 읽기 실패: {json_path} ({e})")
        return set()
    return {Path(im.get("file_name", "")).name
            for im in data.get("images", []) if im.get("file_name")}


def organize_coco(name, src_root, folders, exclude=()):
    train_dir, test_dir, ann_train, ann_test = folders
    base = DATA_ROOT / name
    counts = defaultdict(int)

    jsons = list(walk_files(src_root, {".json"}, exclude))
    train_json = next((j for j in jsons if split_of(j.name) == "train"), None)
    test_json = next((j for j in jsons if split_of(j.name) == "test"), None)

    ann_map = {}
    if train_json:
        ann_map.update({n: "train" for n in coco_basenames(train_json)})
        place(train_json, base / "annotations" / ann_train)
    else:
        print(f"    [warn] train json 을 못 찾음 → 경로 문자열로 폴백")
    if test_json:
        ann_map.update({n: "test" for n in coco_basenames(test_json)})
        place(test_json, base / "annotations" / ann_test)
    else:
        print(f"    [warn] test json 을 못 찾음 → 경로 문자열로 폴백")

    for img in walk_files(src_root, IMG_EXTS, exclude):
        sp = ann_map.get(img.name) or split_of(str(img))
        if sp == "train":
            dst = base / "images" / train_dir / img.name
        elif sp == "test":
            dst = base / "images" / test_dir / img.name
        else:
            counts["unrouted"] += 1
            continue
        counts[place(img, dst)] += 1
    return dict(counts)


def organize_suodac(name, src_root):
    base = DATA_ROOT / name
    counts = defaultdict(int)
    for img in walk_files(src_root, IMG_EXTS):
        # VOC2007/JPEGImages 안의 재복사본은 무시하고 typeN 폴더만 채택
        # (type 폴더에 없고 VOC 에만 있는 경우 대비: VOC 도 살릴지 결정 필요 → 아래 주석)
        type_part = next((p for p in img.parts
                          if p.lower().startswith("type") and len(p) > 4 and p[4:5].isdigit()),
                         None)
        if not type_part:
            counts["skipped_non_type"] += 1  # VOC2007/JPEGImages 등
            continue
        counts[place(img, base / type_part.lower() / img.name)] += 1
    return dict(counts)


HANDLERS = {
    "RUOD": lambda s: organize_coco(
        "RUOD", s,
        ("train", "test", "instances_train.json", "instances_test.json"),
        exclude=("environ",)),   # Environment_pic / Environmet_ANN 제외
    "DUO": lambda s: organize_coco(
        "DUO", s,
        ("train2017", "test2017",
         "instances_train2017.json", "instances_test2017.json")),
    "S-UODAC2020": lambda s: organize_suodac("S-UODAC2020", s),
}


def count_final_images(name):
    """
    organize 후 실제로 '목적지'에 놓인 이미지 수만 센다.
    주의: 원본 폴더(RUOD_pic, DUO/DUO 등)가 같은 data/<name> 아래 남아있으면
    전체 rglob 은 원본+복사본을 이중계산한다. 그래서 목적지 서브트리만 센다.
      - RUOD/DUO: data/<name>/images/ 아래만
      - S-UODAC : data/<name>/typeN/ (최상위) 아래만  (원본은 data/<name>/S-UODAC2020/... 로 더 깊음)
    """
    base = DATA_ROOT / name
    if not base.exists():
        return 0
    if name in ("RUOD", "DUO"):
        target = base / "images"
        if not target.exists():
            return 0
        return sum(1 for p in target.rglob("*")
                   if p.is_file() and p.suffix.lower() in IMG_EXTS)
    if name == "S-UODAC2020":
        # 최상위 typeN 폴더만 (원본 재복사본 data/S-UODAC2020/S-UODAC2020/... 제외)
        total = 0
        for i in range(1, 11):
            d = base / f"type{i}"
            if d.exists():
                total += sum(1 for p in d.iterdir()
                             if p.is_file() and p.suffix.lower() in IMG_EXTS)
        return total
    return sum(1 for p in base.rglob("*")
               if p.is_file() and p.suffix.lower() in IMG_EXTS)


def make_datasets_yaml(processed):
    """검증2용 datasets.yaml 을 실제 생성된 경로 기준으로 작성."""
    lines = ["# 자동 생성됨 — 검증2용. 실제 organize 결과 경로 기준.\n"]

    if "RUOD" in processed:
        lines.append("RUOD_all:     data/RUOD/images/train/*.jpg")
    if "DUO" in processed:
        lines.append("DUO:          data/DUO/images/train2017/*.jpg")
    # Brackish (Roboflow YOLO 버전을 직접 배치했다면)
    lines.append("Brackish:     data/Brackish/train/images/*.jpg   # Roboflow YOLO 버전 경로 확인")
    if "S-UODAC2020" in processed:
        for i in range(1, 8):
            d = DATA_ROOT / "S-UODAC2020" / f"type{i}"
            if d.exists():
                lines.append(f"SUODAC_type{i}: data/S-UODAC2020/type{i}/*.jpg")

    out = Path("datasets.yaml")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[make-yaml] datasets.yaml 생성 완료 ({out.resolve()})")
    print("  → Brackish 경로는 Roboflow 버전 실제 구조에 맞게 한 번 확인하세요.")


def main():
    global MOVE, DRY
    ap = argparse.ArgumentParser()
    ap.add_argument("--move", action="store_true", help="복사 대신 이동")
    ap.add_argument("--dry-run", action="store_true", help="실제 조작 없이 미리보기")
    ap.add_argument("--only", nargs="*", help="특정 데이터셋만")
    ap.add_argument("--make-yaml", action="store_true", help="organize 후 datasets.yaml 생성")
    args = ap.parse_args()
    MOVE, DRY = args.move, args.dry_run

    processed = []
    for name in (args.only or list(SOURCES)):
        src = Path(SOURCES[name]).expanduser()
        print(f"\n=== {name} ===  (src: {src})")
        if not src.exists():
            print(f"  ! 원본 경로 없음. SOURCES['{name}'] 확인 필요.")
            continue
        counts = HANDLERS[name](src)
        summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "처리한 파일 없음"
        print(f"  → {summary}")

        # 검산 (dry-run 아닐 때만 실제 파일 카운트)
        if not DRY:
            final = count_final_images(name)
            exp = EXPECTED.get(name)
            tag = ""
            if exp:
                diff = final - exp
                tag = f"  (기대 {exp}, 차이 {diff:+d})" if diff else f"  (기대 {exp}, 일치 ✓)"
            print(f"  최종 이미지 수: {final}{tag}")
        processed.append(name)

    # 충돌 경고
    if _collisions:
        print(f"\n[!] 파일명 충돌 {len(_collisions)}건 — 같은 목적지에 서로 다른 원본:")
        for s, d in _collisions[:10]:
            print(f"    {s}\n      → {d}")
        if len(_collisions) > 10:
            print(f"    ... 외 {len(_collisions) - 10}건")
        print("    (train/test 에 동일 파일명이 있거나, 중복 소스가 있을 수 있음)")

    if args.make_yaml and not DRY:
        make_datasets_yaml(processed)

    print("\n완료." + ("  (dry-run: 실제로는 아무것도 안 옮김)" if DRY else ""))


if __name__ == "__main__":
    main()