from ultralytics.data.converter import convert_coco
from pathlib import Path
import shutil

SRC = Path("data/RUOD")
TMP = Path("data/_ruod_tmp")

convert_coco(
    labels_dir=str(SRC / "RUOD_ANN"),
    save_dir=str(TMP),
    use_segments=False,
    cls91to80=False,
)

for split in ["train", "test"]:
    (SRC / split / "images").mkdir(parents=True, exist_ok=True)
    (SRC / split / "labels").mkdir(parents=True, exist_ok=True)

    for img in (SRC / "RUOD_pic" / split).glob("*.jpg"):
        shutil.copy(img, SRC / split / "images" / img.name)

    for txt in (TMP / "labels" / split).glob("*.txt"):
        shutil.copy(txt, SRC / split / "labels" / txt.name)

    n_img = len(list((SRC / split / "images").glob("*.jpg")))
    n_lbl = len(list((SRC / split / "labels").glob("*.txt")))
    print(f"{split}: 이미지 {n_img}개, 라벨 {n_lbl}개")

print("RUOD 변환 완료")