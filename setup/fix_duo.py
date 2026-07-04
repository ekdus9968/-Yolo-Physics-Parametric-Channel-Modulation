# fix_duo.py
from pathlib import Path
import shutil

SRC = Path("data/DUO")
DST = Path("data/duo")
TMP = Path("data/_duo_tmp")

for split in ["train", "test"]:
    (DST / split / "images").mkdir(parents=True, exist_ok=True)
    (DST / split / "labels").mkdir(parents=True, exist_ok=True)

    # 이미지 복사: data/DUO/images/train -> data/duo/train/images
    for img in (SRC / "images" / split).glob("*.jpg"):
        shutil.copy(img, DST / split / "images" / img.name)

    # 라벨 복사: data/_duo_tmp/labels/train -> data/duo/train/labels
    for txt in (TMP / "labels" / split).glob("*.txt"):
        shutil.copy(txt, DST / split / "labels" / txt.name)

    n_img = len(list((DST / split / "images").glob("*.jpg")))
    n_lbl = len(list((DST / split / "labels").glob("*.txt")))
    print(f"{split}: 이미지 {n_img}개, 라벨 {n_lbl}개")

print("DUO 재배치 완료")