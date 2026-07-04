# convert_duo.py
from ultralytics.data.converter import convert_coco
from pathlib import Path
import shutil

SRC = Path("data/DUO")                    # 원본 루트
DST = Path("data/duo")                    # YOLO용 출력 루트

# 1) COCO json → YOLO txt 라벨 생성
convert_coco(
    labels_dir=str(SRC / "annotations"),  # instances_train.json, instances_test.json 위치
    save_dir="data/_duo_tmp",
    use_segments=False,
    cls91to80=False,
)
# → data/_duo_tmp/labels/instances_train/*.txt, .../instances_test/*.txt 생성됨

# 2) YOLO 표준 폴더로 재배치
for split_json, split in [("instances_train", "train"), ("instances_test", "test")]:
    (DST / split / "images").mkdir(parents=True, exist_ok=True)
    (DST / split / "labels").mkdir(parents=True, exist_ok=True)

    # 이미지 복사
    img_src = SRC / "images" / split       # DUO/images/train, DUO/images/test
    for img in img_src.glob("*.jpg"):
        shutil.copy(img, DST / split / "images" / img.name)

    # 라벨 복사
    lbl_src = Path("data/_duo_tmp/labels") / split_json
    for txt in lbl_src.glob("*.txt"):
        shutil.copy(txt, DST / split / "labels" / txt.name)

print("DUO 변환 완료")