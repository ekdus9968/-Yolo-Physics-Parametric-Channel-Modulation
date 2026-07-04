from ultralytics.data.converter import convert_coco
from pathlib import Path
import shutil

SRC = Path("data/S-UODAC2020")
TMP = Path("data/_suodac_tmp")
JPEG = SRC / "VOC2007" / "JPEGImages"

convert_coco(
    labels_dir=str(SRC / "COCO_Annotations"),
    save_dir=str(TMP),
    use_segments=False,
    cls91to80=False,
)

mapping = [("source", "train"), ("target", "test")]

for folder_name, split in mapping:
    (SRC / split / "images").mkdir(parents=True, exist_ok=True)
    (SRC / split / "labels").mkdir(parents=True, exist_ok=True)

    lbl_dir = TMP / "labels" / folder_name
    copied_lbl, copied_img, missing = 0, 0, 0

    for txt in lbl_dir.glob("*.txt"):
        shutil.copy(txt, SRC / split / "labels" / txt.name)
        copied_lbl += 1

        img = JPEG / (txt.stem + ".jpg")
        if img.exists():
            shutil.copy(img, SRC / split / "images" / img.name)
            copied_img += 1
        else:
            missing += 1

    print(f"{split}: 라벨 {copied_lbl}개, 이미지 {copied_img}개, 이미지없음 {missing}개")

print("S-UODAC2020 변환 완료")