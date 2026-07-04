"""
train_ppcm_pathB.py — 길 B: 커스텀 학습 루프 + PPCM
====================================================
로더: ultralytics build_yolo_dataset/dataloader 재활용 (augment off)
모델: PPCMDetectionModel (PPCM 이 forward 에 직접 박힘 — hook 아님)
loss: model(batch) 가 직접 반환 (ultralytics 내장 loss)
depth: batch['im_file'] 로 캐시 로드 → model.ppcm_prepare(img01, depth)
검증: --verify 로 1~2 배치만 돌려 PPCM 작동/gradient 확인 후 종료

4 구성: --stage1 --stage2
사용:
  # 먼저 검증 (유령버그 방지)
  python train_ppcm_pathB.py --data data/DUO/data.yaml --stage1 0 --stage2 1 \\
         --dataset-name DUO --verify
  # 실제 학습
  python train_ppcm_pathB.py --data data/DUO/data.yaml --stage1 0 --stage2 1 \\
         --dataset-name DUO --epochs 12
"""

import argparse
import os
import math
import torch
import torch.nn as nn
from torch import optim

from ppcm_yolo_model import PPCMDetectionModel
from train_ppcm import DepthCache


def load_data_yaml(path):
    """data.yaml 파싱 (ultralytics 형식)."""
    import yaml
    with open(path, encoding="utf-8") as f:
        d = yaml.safe_load(f)
    return d


def build_loader(data_yaml, split, imgsz, batch, workers, stride=32,
                  depth_cache_root=None, dataset_name=None):
    """ultralytics 데이터로더 재활용. augment OFF. depth 를 worker에서 미리 로드."""
    from ultralytics.data.build import build_dataloader
    from ultralytics.cfg import get_cfg
    from ultralytics.utils import DEFAULT_CFG
    from depth_dataset import YOLODatasetWithDepth

    cfg = get_cfg(DEFAULT_CFG)
    cfg.mosaic = 0.0; cfg.mixup = 0.0; cfg.copy_paste = 0.0; cfg.cutmix = 0.0
    cfg.hsv_h = 0.0; cfg.hsv_s = 0.0; cfg.hsv_v = 0.0
    cfg.degrees = 0.0; cfg.shear = 0.0; cfg.perspective = 0.0
    cfg.translate = 0.0; cfg.scale = 0.0; cfg.fliplr = 0.0; cfg.flipud = 0.0
    cfg.erasing = 0.0
    cfg.imgsz = imgsz

    d = load_data_yaml(data_yaml)
    root = d.get("path", ".")
    img_path = os.path.join(root, d[split]) if not os.path.isabs(d[split]) else d[split]
    mode = "train" if split == "train" else "val"

    dataset = YOLODatasetWithDepth(
        img_path=img_path, data=d, task="detect", imgsz=imgsz,
        augment=(mode == "train"), hyp=cfg, rect=False, batch_size=batch,
        stride=stride, pad=0.0 if mode == "train" else 0.5,
        single_cls=False, classes=None, fraction=1.0,
        depth_cache_root=depth_cache_root, dataset_name=dataset_name,
    )
    loader = build_dataloader(dataset, batch, workers,
                              shuffle=(mode == "train"), rank=-1)
    return loader, d


def preprocess(batch, device):
    """ultralytics preprocess: GPU 이동 + img/255 → [0,1]."""
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device, non_blocking=(device == "cuda"))
    batch["img"] = batch["img"].float() / 255.0
    return batch


def inject_depth(model, batch, device):
    """batch['depth'] 는 dataloader worker 에서 이미 로드됨."""
    if "depth" not in batch:
        return
    depth = batch["depth"].to(device)
    img01 = batch["img"]
    model.ppcm_prepare(img01, depth)


def build_model(cfg_yaml, nc, s1, s2, weights, device,
                depth_min=0.5, depth_max=10.0, n_transmission=2):
    """PPCMDetectionModel 생성 + pretrained weight 로드."""
    model = PPCMDetectionModel(cfg_yaml, nc=nc, verbose=False,
                               stage1_on=bool(s1), stage2_on=bool(s2),
                               learn_gphi=False,
                               n_transmission=n_transmission,
                               depth_min=depth_min, depth_max=depth_max)
    # pretrained weight 로드 (PPCM 외 레이어). 크기 맞는 것만.
    if weights and os.path.exists(weights):
        from ultralytics import YOLO
        pre = YOLO(weights).model
        pre_sd = pre.state_dict()
        model_sd = model.state_dict()
        # 크기가 일치하는 키만 필터 (COCO 80cls head vs DUO 4cls head 불일치 제외)
        filtered = {k: v for k, v in pre_sd.items()
                    if k in model_sd and v.shape == model_sd[k].shape}
        skipped = [k for k in pre_sd
                   if k in model_sd and pre_sd[k].shape != model_sd[k].shape]
        model.load_state_dict(filtered, strict=False)
        print(f"[model] pretrained 로드: {weights} "
              f"({len(filtered)}/{len(pre_sd)} 레이어)")
        if skipped:
            print(f"[model]   크기 불일치로 새 init: {len(skipped)}개 "
                  f"(Detect head 클래스 레이어 — 정상)")
    model.to(device)
    # loss 기준 초기화
    model.args = _make_hyp()
    return model


def _make_hyp():
    """model.loss 가 참조하는 하이퍼파라미터."""
    from types import SimpleNamespace
    return SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)


# ---------------------------------------------------------------
# 검증 모드: 유령버그 방지 (학습 전 필수)
# ---------------------------------------------------------------
def verify_mode(model, loader, device, s1, s2):
    print("\n" + "="*60)
    print("검증 모드: PPCM 이 실제로 작동/학습되는지 1~2 배치로 확인")
    print("="*60)
    model.train()
    it = iter(loader)
    batch = next(it)
    batch = preprocess(batch, device)
    inject_depth(model, batch, device)

    # baseline(PPCM off) 출력과 비교하기 위해 β/z 백업
    # loss 계산
    loss, loss_items = model.loss(batch)
    loss = loss.sum() if loss.dim() > 0 else loss
    print(f"[검증] loss 계산 성공: {float(loss):.4f}")
    print(f"       loss_items(box,cls,dfl): {[round(float(x),4) for x in loss_items]}")

    # backward → PPCM 파라미터 gradient 확인
    loss.backward()
    checks = []
    if s2 and model.ppcm.stage2 is not None:
        gg = model.ppcm.stage2.gammas[0].grad
        cg = next(model.ppcm.stage2.convs[0].parameters()).grad
        checks.append(("Stage2 γ.grad", gg is not None and gg.abs().item() > 0))
        checks.append(("Stage2 Conv.grad", cg is not None))
    if s1 and model.ppcm.stage1 is not None:
        ag = model.ppcm.stage1.alpha.grad
        checks.append(("Stage1 α.grad", ag is not None))

    print("\n[검증] PPCM 파라미터 gradient:")
    all_ok = True
    for name, ok in checks:
        print(f"   {name}: {'OK' if ok else 'FAIL ✗'}")
        all_ok = all_ok and ok

    # γ=0 identity 확인 (stage2)
    if s2:
        model.eval()
        with torch.no_grad():
            inject_depth(model, batch, device)
            out_on = model(batch["img"])
            # PPCM 끄고
            model.stage2_on = False
            out_off = model(batch["img"])
            model.stage2_on = True
        o_on = out_on[0] if isinstance(out_on,(list,tuple)) else out_on
        o_off = out_off[0] if isinstance(out_off,(list,tuple)) else out_off
        # γ=0(초기)이면 거의 같아야
        diff = (o_on - o_off).abs().max().item()
        print(f"\n[검증] γ=0 초기 상태 PPCM on/off 차이: {diff:.6f} (0 근처면 identity 정상)")

    print("\n" + "="*60)
    print(f"검증 결과: {'통과 — 학습 진행 가능' if all_ok else 'FAIL — 배선 확인 필요'}")
    print("="*60)
    return all_ok


# ---------------------------------------------------------------
# 학습 루프
# ---------------------------------------------------------------
def train(model, train_loader, device, epochs, lr, save_dir, cfg_name):
    os.makedirs(save_dir, exist_ok=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = optim.AdamW(params, lr=lr, weight_decay=5e-4)

    use_amp = (device == "cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    n_batches = len(train_loader)
    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for i, batch in enumerate(train_loader):
            batch = preprocess(batch, device)
            inject_depth(model, batch, device)

            opt.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_amp):
                loss, loss_items = model.loss(batch)
                loss = loss.sum() if loss.dim() > 0 else loss

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += float(loss)
            if (i + 1) % 10 == 0:
                extra = ""
                if model.ppcm.stage2 is not None:
                    gs = [round(float(g),4) for g in model.ppcm.stage2.gammas]
                    extra += f" γ={gs}"
                if model.ppcm.stage1 is not None:
                    extra += f" α={float(model.ppcm.stage1.alpha):.4f}"
                print(f"  ep{epoch} [{i+1}/{n_batches}] loss={running/(i+1):.4f}{extra}")

        avg = running / n_batches
        print(f"[epoch {epoch}] avg_loss={avg:.4f}")
        ckpt = os.path.join(save_dir, f"{cfg_name}_epoch{epoch}.pt")
        torch.save({"model_state_dict": model.state_dict(),
                    "epoch": epoch, "loss": avg}, ckpt)
        if avg < best_loss:
            best_loss = avg
            torch.save({"model_state_dict": model.state_dict(),
                        "epoch": epoch, "loss": avg},
                       os.path.join(save_dir, f"{cfg_name}_best.pt"))
    print(f"\n학습 완료. 체크포인트: {save_dir}/{cfg_name}_best.pt")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", default="yolov8m.pt")
    ap.add_argument("--model-yaml", default="yolov8m.yaml")
    ap.add_argument("--stage1", type=int, default=0, choices=[0,1])
    ap.add_argument("--stage2", type=int, default=0, choices=[0,1])
    ap.add_argument("--dataset-name", default="DUO")
    ap.add_argument("--depth-cache", default="depth_cache")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.00125)
    ap.add_argument("--depth-min", type=float, default=0.5)
    ap.add_argument("--depth-max", type=float, default=10.0)
    ap.add_argument("--n-transmission", type=int, default=2)
    ap.add_argument("--save-dir", default="runs/ppcm_pathB")
    ap.add_argument("--verify", action="store_true", help="1~2배치 검증 후 종료")
    args = ap.parse_args()

    cfg_name = {(0,0):"baseline",(1,0):"stage1",(0,1):"stage2",(1,1):"full"}[
        (args.stage1, args.stage2)]
    print(f"=== 길B PPCM | config={cfg_name} | dataset={args.dataset_name} ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = load_data_yaml(args.data)
    nc = d.get("nc", len(d.get("names", {})))

    # 로더 (depth 를 worker 에서 로드하도록 인자 전달 — stage on 일 때)
    dcr = args.depth_cache if (args.stage1 or args.stage2) else None
    train_loader, _ = build_loader(args.data, "train", args.imgsz,
                                   args.batch, args.workers,
                                   depth_cache_root=dcr,
                                   dataset_name=args.dataset_name)

    # 모델
    model = build_model(args.model_yaml, nc, args.stage1, args.stage2,
                        args.weights, device,
                        depth_min=args.depth_min, depth_max=args.depth_max,
                        n_transmission=args.n_transmission)

    # depth 는 로더(worker)에서 batch['depth'] 로 로드됨.
    if args.stage1 or args.stage2:
        print(f"[depth] cache(worker): {args.depth_cache}/{args.dataset_name}")

    if args.verify:
        verify_mode(model, train_loader, device, args.stage1, args.stage2)
        return

    save_dir = os.path.join(args.save_dir, f"{args.dataset_name}_{cfg_name}")
    train(model, train_loader, device, args.epochs, args.lr, save_dir, cfg_name)


if __name__ == "__main__":
    main()