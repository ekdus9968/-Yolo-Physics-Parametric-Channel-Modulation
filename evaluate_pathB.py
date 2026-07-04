"""
evaluate_pathB.py — PPCMDetectionModel 평가 (pycocotools, AP_small 분리)
=======================================================================
길 B 커스텀 모델(PPCM forward 에 박힘) 전용 평가.
COCO GT + 예측 → mAP@50, mAP@[.5:.95], AP_small/medium/large + 클래스별.

핵심:
  - 체크포인트(model_state_dict, PPCM 포함) 로드
  - val 이미지마다 depth 캐시 로드 → ppcm_prepare → 추론 → NMS
  - pycocotools 로 COCO 지표 (AP_small 분리 = 물리 모듈 스토리 핵심)

사용:
  python evaluate_pathB.py --ckpt runs/ppcm_pathB/DUO_stage2/stage2_best.pt \\
      --data data/DUO/data.yaml --ann data/DUO/annotations/instances_test.json \\
      --stage1 0 --stage2 1 --dataset-name DUO --depth-cache depth_cache

  # 4구성 한번에 (ablation):
  python evaluate_pathB.py --ablation --ckpt-dir runs/ppcm_pathB \\
      --data data/DUO/data.yaml --ann data/DUO/annotations/instances_test.json \\
      --dataset-name DUO --depth-cache depth_cache
"""

import argparse
import os
import json
import numpy as np
import torch

from ppcm_yolo_model import PPCMDetectionModel
from train_ppcm import DepthCache

STAT = {"mAP": 0, "mAP50": 1, "mAP75": 2,
        "AP_small": 3, "AP_medium": 4, "AP_large": 5}


def load_data_yaml(path):
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model_for_eval(cfg_yaml, nc, s1, s2, ckpt_path, device,
                         depth_min=0.5, depth_max=10.0, n_transmission=2):
    model = PPCMDetectionModel(cfg_yaml, nc=nc, verbose=False,
                               stage1_on=bool(s1), stage2_on=bool(s2),
                               learn_gphi=False, n_transmission=n_transmission,
                               depth_min=depth_min, depth_max=depth_max)
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return model


@torch.no_grad()
def predict_image(model, img_path, depth_cache, device, imgsz, conf, iou, nc):
    """한 이미지 추론 → NMS 후 (boxes_xyxy, scores, labels) (원본 좌표계)."""
    import cv2
    from ultralytics.utils.nms import non_max_suppression
    from ultralytics.utils.ops import scale_boxes

    im0 = cv2.imread(img_path)
    h0, w0 = im0.shape[:2]
    # letterbox (ultralytics 방식 간이): resize 유지비율 + pad
    from ultralytics.data.augment import LetterBox
    lb = LetterBox((imgsz, imgsz), auto=False, stride=32)
    im = lb(image=im0)
    im = im[:, :, ::-1].transpose(2, 0, 1).copy()   # BGR→RGB, HWC→CHW
    im = torch.from_numpy(im).float().unsqueeze(0).to(device) / 255.0

    # depth 주입 (stage on 이면)
    if depth_cache is not None:
        d = depth_cache.get(img_path, target_hw=im.shape[-2:]).unsqueeze(0).to(device)
        model.ppcm_prepare(im, d)

    out = model(im)
    preds = out[0] if isinstance(out, (list, tuple)) else out
    preds = non_max_suppression(preds, conf, iou, max_det=300)[0]  # [N,6] xyxy,conf,cls

    if preds is None or len(preds) == 0:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)

    # letterbox 좌표 → 원본 좌표
    preds[:, :4] = scale_boxes(im.shape[2:], preds[:, :4], (h0, w0))
    boxes = preds[:, :4].cpu().numpy()
    scores = preds[:, 4].cpu().numpy()
    labels = preds[:, 5].cpu().numpy().astype(int)
    return boxes, scores, labels


def evaluate(model, ann_file, img_dir, depth_cache, device,
             imgsz=640, conf=0.001, iou=0.7, nc=4, cat_ids=None):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(ann_file)
    img_ids = coco_gt.getImgIds()
    # YOLO 클래스 인덱스(0..nc-1) → COCO category_id 매핑
    if cat_ids is None:
        cat_ids = sorted(coco_gt.getCatIds())
    idx_to_catid = {i: cid for i, cid in enumerate(cat_ids)}

    results = []
    from tqdm import tqdm
    for img_id in tqdm(img_ids, desc="eval"):
        info = coco_gt.imgs[img_id]
        img_path = os.path.join(img_dir, info["file_name"])
        if not os.path.exists(img_path):
            continue
        boxes, scores, labels = predict_image(
            model, img_path, depth_cache, device, imgsz, conf, iou, nc)
        for (x1, y1, x2, y2), sc, lb in zip(boxes, scores, labels):
            results.append({
                "image_id": int(img_id),
                "category_id": int(idx_to_catid.get(int(lb), int(lb))),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(sc),
            })

    if not results:
        print("  검출 없음")
        return None

    coco_dt = coco_gt.loadRes(results)
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.evaluate(); ev.accumulate(); ev.summarize()

    stats = {k: float(ev.stats[i] * 100) for k, i in STAT.items()}

    # 클래스별 AP@50 (scallop 등 소수 클래스 추적)
    per_class = {}
    # COCOeval precision: [T,R,K,A,M]; T=iou, K=class
    prec = ev.eval["precision"]  # [10,101,K,4,3]
    for k, cid in enumerate(cat_ids):
        p = prec[0, :, k, 0, 2]  # iou=0.5, area=all, maxdet=100
        p = p[p > -1]
        ap50 = float(p.mean() * 100) if p.size else 0.0
        name = coco_gt.cats[cid]["name"]
        per_class[name] = ap50
    stats["per_class_AP50"] = per_class
    return stats


def print_stats(label, s):
    print(f"\n[{label}]")
    print(f"  mAP@50={s['mAP50']:.2f}  mAP@[.5:.95]={s['mAP']:.2f}")
    print(f"  AP_small={s['AP_small']:.2f}  AP_medium={s['AP_medium']:.2f}  "
          f"AP_large={s['AP_large']:.2f}")
    if "per_class_AP50" in s:
        cls_str = "  ".join(f"{n}={v:.1f}" for n, v in s["per_class_AP50"].items())
        print(f"  per-class AP50: {cls_str}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ann", required=True, help="COCO GT json (test)")
    ap.add_argument("--img-dir", default=None, help="test 이미지 폴더(생략시 data.yaml)")
    ap.add_argument("--model-yaml", default="yolov8m.yaml")
    ap.add_argument("--dataset-name", default="DUO")
    ap.add_argument("--depth-cache", default="depth_cache")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--n-transmission", type=int, default=2)
    # 단일 평가
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--stage1", type=int, default=0)
    ap.add_argument("--stage2", type=int, default=0)
    # ablation (4구성)
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--ckpt-dir", default="runs/ppcm_pathB")
    ap.add_argument("--out", default="ablation_pathB.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = load_data_yaml(args.data)
    nc = d.get("nc", len(d.get("names", {})))
    img_dir = args.img_dir
    if img_dir is None:
        root = d.get("path", ".")
        val = d.get("test", d.get("val"))
        img_dir = os.path.join(root, val) if not os.path.isabs(val) else val

    configs = ([("baseline",0,0),("stage1",1,0),("stage2",0,1),("full",1,1)]
               if args.ablation else
               [({(0,0):"baseline",(1,0):"stage1",(0,1):"stage2",(1,1):"full"}[
                   (args.stage1,args.stage2)], args.stage1, args.stage2)])

    all_res = {}
    for cfg, s1, s2 in configs:
        if args.ablation:
            ckpt = os.path.join(args.ckpt_dir, f"{args.dataset_name}_{cfg}",
                                f"{cfg}_best.pt")
        else:
            ckpt = args.ckpt
        if not ckpt or not os.path.exists(ckpt):
            print(f"[skip] {cfg}: 체크포인트 없음 ({ckpt})")
            continue
        print(f"\n{'='*50}\n평가: {cfg}  ({ckpt})")
        model = build_model_for_eval(args.model_yaml, nc, s1, s2, ckpt, device,
                                     n_transmission=args.n_transmission)
        depth_cache = DepthCache(args.depth_cache, args.dataset_name) if (s1 or s2) else None
        s = evaluate(model, args.ann, img_dir, depth_cache, device,
                     imgsz=args.imgsz, conf=args.conf, iou=args.iou, nc=nc)
        if s:
            all_res[cfg] = s
            print_stats(cfg, s)

    # ablation 요약
    if len(all_res) > 1:
        print(f"\n{'='*60}\nAblation Summary\n{'='*60}")
        base = all_res.get("baseline", {}).get("mAP50", 0)
        print(f"{'config':<12}{'mAP50':>8}{'Δ':>7}{'AP_s':>8}{'AP_m':>8}{'AP_l':>8}")
        for cfg, s in all_res.items():
            dd = s["mAP50"] - base
            print(f"{cfg:<12}{s['mAP50']:>8.2f}{dd:>+7.2f}"
                  f"{s['AP_small']:>8.2f}{s['AP_medium']:>8.2f}{s['AP_large']:>8.2f}")
        # scallop(소수클래스) 별도 추적
        print("\n[소수 클래스 AP50 변화]")
        if "baseline" in all_res:
            for name in all_res["baseline"].get("per_class_AP50", {}):
                bl = all_res["baseline"]["per_class_AP50"][name]
                row = f"  {name:12}: baseline={bl:.1f}"
                for cfg in ("stage1","stage2","full"):
                    if cfg in all_res:
                        v = all_res[cfg]["per_class_AP50"].get(name, 0)
                        row += f"  {cfg}={v:.1f}({v-bl:+.1f})"
                print(row)

    with open(args.out, "w") as f:
        json.dump(all_res, f, indent=2)
    print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()