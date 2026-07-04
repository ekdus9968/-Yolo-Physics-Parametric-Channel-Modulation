# convert_ruod.py
from ultralytics.data.converter import convert_coco
from pathlib import Path
import shutil

SRC = Path("data/RUOD")
TMP = Path("data/_ruod_tmp")

# 1) COCO json -> YOLO txt 라벨 생성
convert_coco(
    labels_dir=str(SRC / "RUOD_ANN"),   # instances_train.json, instances_test.json 위치
    save_dir=str(TMP),
    use_segments=False,
    cls91to80=False,
)

# 2) YOLO 표준 폴더로 재배치 (원본 RUOD 폴더 안에)
for split in ["train", "test"]:
    (SRC / split / "images").mkdir(parents=True, exist_ok=True)
    (SRC / split / "labels").mkdir(parents=True, exist_ok=True)

    # 이미지 복사: data/RUOD/RUOD_pic/train -> data/RUOD/train/images
    for img in (SRC / "RUOD_pic" / split).glob("*.jpg"):
        shutil.copy(img, SRC / split / "images" / img.name)

    # 라벨 복사: data/_ruod_tmp/labels/train -> data/RUOD/train/labels
    for txt in (TMP / "labels" / split).glob("*.txt"):
        shutil.copy(txt, SRC / split / "labels" / txt.name)

    n_img = len(list((SRC / split / "images").glob("*.jpg")))
    n_lbl = len(list((SRC / split / "labels").glob("*.txt")))
    print(f"{split}: 이미지 {n_img}개, 라벨 {n_lbl}개")

print("RUOD 변환 완료")