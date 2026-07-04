"""
train_ppcm_yolo.py — 경로 B: ultralytics .train() + PPCM
========================================================
결정 반영:
  - 경로 B (ultralytics .train())
  - augment off (mosaic=0, hsv/색 off) → 파일명 기반 depth 매칭 유효
  - depth 캐싱 (augment off 라 원본=학습이미지)
  - g_φ Mode 1 (휴리스틱)
  - num_classes 는 data yaml 에서
  - 4 토글: --stage1 --stage2

핵심 배선:
  ultralytics 콜백 on_train_batch_start / on_val_batch_start 에서
  batch['im_file'] 로 depth 를 캐시 로드 → ppcm.prepare(img, depth).
  그 뒤 model forward 시 hook 이 Stage1/Stage2 자동 적용.

  im_file 이 collate_fn 을 통과해 batch 에 남는 것을 확인함(ultralytics 8.4.x).

주의: augment off 가 전제. mosaic/색 augment 를 켜면 원본↔depth 정합이 깨져
      이 방식이 무효가 된다. (그 경우 augment 후 depth 생성 필요 = 느림)
"""

import argparse
import os
import torch
from ppcm_modules import PPCM, attach_ppcm_hooks
from train_ppcm import DepthCache, build_ppcm_for_model


def make_ppcm_callbacks(ppcm, depth_cache, device):
    """
    ultralytics YOLO 에 등록할 콜백들.
    배치 시작 시 im_file 로 depth 로드 → ppcm.prepare.
    """
    def _prepare_from_batch(trainer_or_validator):
        batch = getattr(trainer_or_validator, "batch", None)
        if batch is None or "im_file" not in batch:
            return
        im_files = batch["im_file"]
        imgs = batch["img"]                      # [B,3,H,W], 0-255 or 0-1
        if imgs.dtype == torch.uint8:
            img01 = imgs.float() / 255.0
        else:
            img01 = imgs.float()
            if img01.max() > 1.5:                # 0-255 float
                img01 = img01 / 255.0
        img01 = img01.to(device)
        H, W = img01.shape[-2:]

        # depth 배치 로드 (파일명 매칭)
        ds = []
        for p in im_files:
            d = depth_cache.get(p, target_hw=(H, W))   # (1,H,W)
            ds.append(d)
        depth = torch.stack(ds, 0).to(device)          # [B,1,H,W]

        ppcm.prepare(img01, depth)

    def on_train_batch_start(trainer):
        _prepare_from_batch(trainer)

    def on_val_batch_start(validator):
        _prepare_from_batch(validator)

    return {
        "on_train_batch_start": on_train_batch_start,
        "on_val_batch_start": on_val_batch_start,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="ultralytics data yaml (train/val/nc)")
    ap.add_argument("--weights", default="yolov8m.pt")
    ap.add_argument("--stage1", type=int, default=0, choices=[0, 1])
    ap.add_argument("--stage2", type=int, default=0, choices=[0, 1])
    ap.add_argument("--depth-cache", default="depth_cache")
    ap.add_argument("--dataset-name", default="DUO")
    ap.add_argument("--n-transmission", type=int, default=2, choices=[2, 3])
    ap.add_argument("--depth-min", type=float, default=0.5)
    ap.add_argument("--depth-max", type=float, default=10.0)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--project", default="runs/ppcm")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    from ultralytics import YOLO

    cfg_name = {(0,0):"baseline",(1,0):"stage1",(0,1):"stage2",(1,1):"full"}[
        (args.stage1, args.stage2)]
    run_name = args.name or f"{args.dataset_name}_{cfg_name}"
    print(f"=== PPCM config: {cfg_name} | dataset: {args.dataset_name} ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo = YOLO(args.weights)
    model = yolo.model.to(device)

    # project 를 절대경로로 → ultralytics 가 runs/detect 아래 중첩시키는 것 방지
    project_abs = os.path.abspath(args.project)

    # PPCM 부착 (baseline 이면 무동작, hook 도 identity)
    ppcm = None
    if args.stage1 or args.stage2:
        ppcm, handles, det_idx = build_ppcm_for_model(
            model, bool(args.stage1), bool(args.stage2),
            learn_gphi=False,
            n_transmission=args.n_transmission,
            depth_min=args.depth_min, depth_max=args.depth_max)
        ppcm.to(device)
        depth_cache = DepthCache(args.depth_cache, args.dataset_name)

        # 콜백 등록
        cbs = make_ppcm_callbacks(ppcm, depth_cache, device)
        for event, fn in cbs.items():
            yolo.add_callback(event, fn)

        # PPCM 파라미터를 optimizer 가 학습하도록:
        # ultralytics 는 model.parameters() 를 옵티마이저에 넣으므로,
        # PPCM 을 model 의 서브모듈로 등록해 파라미터가 포함되게 한다.
        model.add_module("ppcm", ppcm)
        print(f"[train] PPCM attached as submodule. "
              f"learnable params: {sum(p.numel() for p in ppcm.parameters() if p.requires_grad)}")

    # ---- augment 완전 OFF (결정 B + depth 정합) ----
    # 주의: fliplr 도 0. flip 을 켜면 batch['img'] 는 반전되지만 depth 캐시는
    #       원본이라 정합이 깨진다. augment off 를 진짜로 지키려면 flip 도 off.
    #       (flip 이득은 나중에 depth 동기 flip 구현 후 별도로)
    train_kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=project_abs,
        name=run_name,
        # --- augmentation 전부 OFF ---
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,   # 색 augment off (물리 보존)
        degrees=0.0, shear=0.0, perspective=0.0,
        translate=0.0, scale=0.0,
        fliplr=0.0,                         # ★ flip 도 off (depth 캐시 정합)
        flipud=0.0,
    )
    print("[train] augmentation: ALL OFF (mosaic/color/flip) — depth 정합 보장")
    print("[train] (depth 캐시는 원본 기준 — augment off 라 정합 완벽 유지)")

    yolo.train(**train_kwargs)

    # hook 정리
    if ppcm is not None:
        for h in handles:
            h.remove()


if __name__ == "__main__":
    main()