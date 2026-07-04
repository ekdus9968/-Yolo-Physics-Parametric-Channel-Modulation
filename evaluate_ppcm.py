"""
evaluate_ppcm.py — PPCM + YOLOv8 4구성 ablation 평가
=====================================================
예전 evaluate.py(Faster R-CNN 기반) 전면 재작성.

버린 것: PPCMPipeline, dict 출력 파싱, img_id 추적 버그, water_types 인자
살린 것: COCOeval 구조, 4-config 순회, ablation summary + json 저장
추가:    YOLOv8 출력 파싱, AP_small/medium/large 분리(우리가 강조), depth 캐시

4 구성:
  baseline (s1=0,s2=0) / stage1 (1,0) / stage2 (0,1) / full (1,1)

평가 지표:
  mAP@50, mAP@50:95, 그리고 ★AP_small / AP_medium / AP_large★
  (물리 모듈 이득은 작은/원거리 객체에 몰리므로 분리 리포트 필수)

사용:
  python evaluate_ppcm.py --data-cfg duo.yaml --ckpt-dir runs/ppcm \\
         --depth-cache depth_cache --dataset-name DUO --num-classes 4
"""

import argparse
import os
import json
import numpy as np
import torch
from tqdm import tqdm

from ppcm_modules import PPCM, attach_ppcm_hooks
from train_ppcm import DepthCache, build_ppcm_for_model


# COCOeval.stats 인덱스 (bbox):
#  [0] AP@[.5:.95]  [1] AP@.5  [2] AP@.75
#  [3] AP_small     [4] AP_medium  [5] AP_large
#  [6..11] AR ...
STAT = {"mAP": 0, "mAP50": 1, "mAP75": 2,
        "AP_small": 3, "AP_medium": 4, "AP_large": 5}


def yolo_predict_to_coco(results, img_id):
    """
    ultralytics 예측 결과(Results) → COCO detection dict 리스트.
    ultralytics Results.boxes: xyxy, conf, cls (모두 tensor).
    """
    out = []
    if results.boxes is None or len(results.boxes) == 0:
        return out
    boxes = results.boxes.xyxy.cpu().numpy()
    scores = results.boxes.conf.cpu().numpy()
    labels = results.boxes.cls.cpu().numpy().astype(int)
    for (x1, y1, x2, y2), sc, lb in zip(boxes, scores, labels):
        out.append({
            "image_id": int(img_id),
            "category_id": int(lb),        # 주의: COCO cat_id 매핑 필요시 아래 remap
            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
            "score": float(sc),
        })
    return out


def evaluate_one(cfg_name, s1, s2, ckpt_path, args, cat_remap=None):
    """한 구성(체크포인트) 평가 → COCOeval stats 반환."""
    from ultralytics import YOLO
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) 모델 로드 (학습된 체크포인트)
    yolo = YOLO(ckpt_path)
    model = yolo.model.to(device).eval()

    # 2) PPCM 부착 (구성에 맞게)
    ppcm = None
    if s1 or s2:
        ppcm, handles, det_idx = build_ppcm_for_model(
            model, bool(s1), bool(s2),
            learn_gphi=False,          # Mode1 (평가 시)
            n_transmission=args.n_transmission,
            depth_min=args.depth_min, depth_max=args.depth_max)
        ppcm.to(device).eval()
        depth_cache = DepthCache(args.depth_cache, args.dataset_name)

    # 3) GT 로드
    coco_gt = COCO(args.ann_file)
    img_ids = coco_gt.getImgIds()

    results = []
    for img_id in tqdm(img_ids, desc=cfg_name):
        info = coco_gt.imgs[img_id]
        img_path = os.path.join(args.img_dir, info["file_name"])
        if not os.path.exists(img_path):
            continue

        # depth 주입 (stage on 일 때)
        if ppcm is not None:
            import cv2
            im = cv2.imread(img_path)
            hw = im.shape[:2]
            depth = depth_cache.get(img_path, target_hw=hw).unsqueeze(0).to(device)
            # 이미지도 [0,1] RGB 로 prepare 에 전달 (g_φ Mode1 휴리스틱용)
            img_rgb = torch.from_numpy(im[:, :, ::-1].copy()).permute(2, 0, 1)
            img_rgb = (img_rgb.float() / 255.0).unsqueeze(0).to(device)
            ppcm.prepare(img_rgb, depth)

        # YOLO 추론 (hook 이 자동으로 PPCM 적용)
        with torch.no_grad():
            res = yolo.predict(img_path, conf=args.conf, iou=args.iou,
                               verbose=False, device=device)[0]

        det = yolo_predict_to_coco(res, img_id)
        if cat_remap:
            for d in det:
                d["category_id"] = cat_remap.get(d["category_id"], d["category_id"])
        results.extend(det)

    if ppcm is not None:
        for h in handles:
            h.remove()

    if not results:
        print(f"  {cfg_name}: no detections")
        return None

    coco_dt = coco_gt.loadRes(results)
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.evaluate(); ev.accumulate(); ev.summarize()
    return {k: float(ev.stats[i] * 100) for k, i in STAT.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann-file", required=True, help="COCO GT json")
    ap.add_argument("--img-dir", required=True, help="평가 이미지 폴더")
    ap.add_argument("--ckpt-dir", required=True, help="4개 체크포인트 폴더")
    ap.add_argument("--depth-cache", default="depth_cache")
    ap.add_argument("--dataset-name", default="DUO")
    ap.add_argument("--num-classes", type=int, default=4)
    ap.add_argument("--n-transmission", type=int, default=2)
    ap.add_argument("--depth-min", type=float, default=0.5)
    ap.add_argument("--depth-max", type=float, default=10.0)
    ap.add_argument("--conf", type=float, default=0.001)  # eval 은 낮게
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--ckpt-pattern", default="{cfg}_best.pt",
                    help="체크포인트 파일명 패턴. {cfg}=baseline/stage1/stage2/full")
    ap.add_argument("--out", default="ablation_results.json")
    args = ap.parse_args()

    print("=" * 60)
    print("PPCM + YOLOv8 — Ablation Evaluation")
    print("=" * 60)

    configs = [
        ("baseline", 0, 0, "Baseline (no PPCM)"),
        ("stage1",   1, 0, "Stage 1 only"),
        ("stage2",   0, 1, "Stage 2 only"),
        ("full",     1, 1, "Full PPCM (S1+S2)"),
    ]

    all_results = {}
    for cfg, s1, s2, label in configs:
        ckpt = os.path.join(args.ckpt_dir, args.ckpt_pattern.format(cfg=cfg))
        if not os.path.exists(ckpt):
            print(f"\n[skip] checkpoint 없음: {ckpt}")
            continue
        print(f"\n{'-'*40}\nEvaluating: {label}")
        stats = evaluate_one(cfg, s1, s2, ckpt, args)
        if stats:
            all_results[label] = stats
            print(f"  mAP@50={stats['mAP50']:.2f}  mAP@[.5:.95]={stats['mAP']:.2f}")
            print(f"  AP_small={stats['AP_small']:.2f}  "
                  f"AP_medium={stats['AP_medium']:.2f}  "
                  f"AP_large={stats['AP_large']:.2f}")

    # ---- Ablation summary ----
    print("\n" + "=" * 60)
    print("Ablation Summary")
    print("=" * 60)
    if not all_results:
        print("결과 없음 (체크포인트 확인).")
        return
    base = all_results.get("Baseline (no PPCM)", {}).get("mAP50", 0.0)
    best_map = max(r["mAP50"] for r in all_results.values())
    print(f"{'config':<22}{'mAP50':>8}{'Δ':>8}{'AP_s':>8}{'AP_m':>8}{'AP_l':>8}")
    for label, r in all_results.items():
        d = r["mAP50"] - base
        mark = " ←best" if r["mAP50"] == best_map else ""
        print(f"{label:<22}{r['mAP50']:>8.2f}{d:>+8.2f}"
              f"{r['AP_small']:>8.2f}{r['AP_medium']:>8.2f}{r['AP_large']:>8.2f}{mark}")

    # ★ 핵심: AP_small 에서의 이득 별도 강조 (물리 모듈 스토리)
    print("\n[해석] 물리 모듈 이득은 AP_small 에 몰릴 것으로 예상.")
    if "Baseline (no PPCM)" in all_results:
        bs = all_results["Baseline (no PPCM)"]["AP_small"]
        for label, r in all_results.items():
            if label == "Baseline (no PPCM)":
                continue
            print(f"  {label}: AP_small Δ = {r['AP_small']-bs:+.2f}")

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()