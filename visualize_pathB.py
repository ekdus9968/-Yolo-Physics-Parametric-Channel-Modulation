"""
visualize_pathB.py — PPCMDetectionModel 시각화 (가능한 것 전부)
==============================================================
커스텀 모델(PPCM forward 박힘)에서 뽑을 수 있는 모든 시각화.

생성 목록:
  A. Stage1 채널 재가중: raw / corrected / 채널 gain / |diff|
  B. Stage2 채널별 transmission (t_R, t_B) + modulation 크기
  C. 학습된 물리 파라미터: γ_l (스케일별), α, 예측 β_D 분포
  D. depth map 시각화
  E. 검출 결과 (GT vs Pred)

체크포인트에서 γ/α 를 읽어 실제 학습된 값으로 시각화.

사용:
  python visualize_pathB.py --ckpt runs/ppcm_pathB/DUO_stage2/stage2_best.pt \\
      --stage1 0 --stage2 1 --dataset-name DUO --depth-cache depth_cache \\
      --img-dir data/DUO/test/images --n-images 4 --out viz_out
"""

import argparse
import os
import glob
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from ppcm_yolo_model import PPCMDetectionModel
from train_ppcm import DepthCache


def _to_img(t):
    """(3,H,W) tensor [0,1] → HWC numpy."""
    t = t.detach().cpu().float()
    return np.clip(t.permute(1, 2, 0).numpy(), 0, 1)


def load_model(cfg_yaml, nc, s1, s2, ckpt, device, n_transmission=2):
    model = PPCMDetectionModel(cfg_yaml, nc=nc, verbose=False,
                               stage1_on=bool(s1), stage2_on=bool(s2),
                               learn_gphi=False, n_transmission=n_transmission)
    c = torch.load(ckpt, map_location=device)
    sd = c["model_state_dict"] if "model_state_dict" in c else c
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return model


def load_image_tensor(img_path, imgsz, device):
    import cv2
    from ultralytics.data.augment import LetterBox
    im0 = cv2.imread(img_path)
    lb = LetterBox((imgsz, imgsz), auto=False, stride=32)
    im = lb(image=im0)
    rgb = im[:, :, ::-1].transpose(2, 0, 1).copy()
    t = torch.from_numpy(rgb).float().unsqueeze(0).to(device) / 255.0
    return t, im0


# ── A. Stage1 시각화 ───────────────────────────────────────
def viz_stage1(model, img_t, out_path):
    beta = model._ppcm_beta                       # [B,3]
    alpha = float(model.ppcm.stage1.alpha)
    corrected = model.ppcm.stage1(img_t, beta)

    raw = _to_img(img_t[0]); corr = _to_img(corrected[0])
    diff = np.abs(raw - corr)
    r = torch.exp(-beta[0]); r_bar = r.mean()
    gain = (1 + alpha * (r - r_bar)).detach().cpu().numpy()

    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    ax[0].imshow(raw); ax[0].set_title(f"Raw\nmean={raw.mean():.3f}"); ax[0].axis("off")
    ax[1].imshow(corr); ax[1].set_title(f"After Stage1\nmean={corr.mean():.3f}"); ax[1].axis("off")
    ax[2].bar(["R","G","B"], gain, color=["red","green","blue"], alpha=0.7)
    ax[2].axhline(1, color="k", ls="--"); ax[2].set_title(
        f"Channel gain (alpha={alpha:.3f})\nbeta=[{beta[0,0]:.2f},{beta[0,1]:.2f},{beta[0,2]:.2f}]")
    im = ax[3].imshow(diff.mean(2), cmap="hot")
    ax[3].set_title(f"|Raw-Corrected|\nmax={diff.max():.4f}"); ax[3].axis("off")
    plt.colorbar(im, ax=ax[3], fraction=0.046)
    plt.suptitle("Stage 1: channel reliability reweighting (residual)")
    plt.tight_layout(); plt.savefig(out_path, dpi=110, bbox_inches="tight"); plt.close()


# ── B. Stage2 시각화 ───────────────────────────────────────
def viz_stage2(model, img_t, out_path):
    z = model._ppcm_z; beta = model._ppcm_beta
    T = model.ppcm.stage2._transmission(z, beta)   # [B,n_t,H,W]
    n_t = T.shape[1]

    fig, ax = plt.subplots(1, n_t + 1, figsize=(6*(n_t+1), 5))
    # depth
    zmap = z[0,0].cpu().numpy()
    im0 = ax[0].imshow(zmap, cmap="plasma"); ax[0].set_title(
        f"Depth z (0=near,1=far)\nmean={zmap.mean():.3f}"); ax[0].axis("off")
    plt.colorbar(im0, ax=ax[0], fraction=0.046)
    # transmission maps
    tnames = ["t_R (red transmission)", "t_B (blue transmission)"] if n_t==2 \
        else ["t_R","t_G","t_B"]
    for j in range(n_t):
        tmap = T[0,j].cpu().numpy()
        im = ax[j+1].imshow(tmap, cmap="viridis", vmin=0, vmax=1)
        ax[j+1].set_title(f"{tnames[j]}\nmean={tmap.mean():.3f} (low=attenuated)")
        ax[j+1].axis("off"); plt.colorbar(im, ax=ax[j+1], fraction=0.046)
    plt.suptitle("Stage 2: per-channel transmission (physics confidence)\n"
                 "t_R drops faster with depth than t_B (red attenuates first)")
    plt.tight_layout(); plt.savefig(out_path, dpi=110, bbox_inches="tight"); plt.close()


# ── C. 학습된 물리 파라미터 ─────────────────────────────────
def viz_params(model, out_path):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    # γ per scale
    if model.ppcm.stage2 is not None:
        gs = [float(g) for g in model.ppcm.stage2.gammas]
        names = [f"N{i+3}\n({s})" for i,s in enumerate(["small","med","large"])]
        colors = ["#e74c3c" if g<0 else "#27ae60" for g in gs]
        ax[0].bar(names, gs, color=colors, alpha=0.8)
        ax[0].axhline(0, color="k", lw=1)
        ax[0].set_title("Learned Stage2 gate gamma per scale\n"
                        "(magnitude = physics contribution at that scale)")
        ax[0].set_ylabel("gamma")
        for i,g in enumerate(gs):
            ax[0].text(i, g, f"{g:+.4f}", ha="center",
                       va="bottom" if g>=0 else "top")
    # α
    if model.ppcm.stage1 is not None:
        a = float(model.ppcm.stage1.alpha)
        ax[1].bar(["alpha"], [a], color="purple", alpha=0.7)
        ax[1].set_title(f"Learned Stage1 strength\nalpha={a:.4f}")
    else:
        ax[1].axis("off"); ax[1].text(0.5,0.5,"Stage1 off",ha="center")
    plt.suptitle("Learned physics parameters (interpretability)")
    plt.tight_layout(); plt.savefig(out_path, dpi=110, bbox_inches="tight"); plt.close()


# ── C2. β_D 분포 (여러 이미지) ──────────────────────────────
def viz_beta_dist(betas, out_path):
    from ppcm_modules import PhysInterp
    anchors = PhysInterp().anchors.numpy()
    names = ["I","II","III","1C","5C","9C"]
    betas = np.asarray(betas)
    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(betas[:,2], betas[:,0], s=20, alpha=0.5, color="steelblue",
               label="predicted beta")
    ax.scatter(anchors[:,2], anchors[:,0], s=200, marker="X", color="crimson",
               edgecolor="black", zorder=5, label="S&M anchors")
    for i,nm in enumerate(names):
        ax.annotate(nm, (anchors[i,2], anchors[i,0]), fontsize=11,
                    fontweight="bold", xytext=(5,5), textcoords="offset points")
    ax.set_xlabel("beta_B (turbidity axis)"); ax.set_ylabel("beta_R")
    ax.set_title("Predicted beta_D vs physical anchors\n"
                 "(clustering near anchors = physically valid water-type)")
    ax.legend(); plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight"); plt.close()


# ── E. 검출 시각화 ─────────────────────────────────────────
def viz_detection(model, img_path, depth_cache, device, imgsz, names,
                  out_path, conf=0.25, iou=0.7):
    from ultralytics.utils.nms import non_max_suppression
    from ultralytics.utils.ops import scale_boxes
    img_t, im0 = load_image_tensor(img_path, imgsz, device)
    h0, w0 = im0.shape[:2]
    if depth_cache is not None:
        d = depth_cache.get(img_path, target_hw=img_t.shape[-2:]).unsqueeze(0).to(device)
        model.ppcm_prepare(img_t, d)
    with torch.no_grad():
        out = model(img_t)
    preds = out[0] if isinstance(out,(list,tuple)) else out
    preds = non_max_suppression(preds, conf, iou, max_det=300)[0]

    fig, ax = plt.subplots(figsize=(10,8))
    ax.imshow(im0[:,:,::-1]); ax.axis("off")
    if preds is not None and len(preds):
        preds[:,:4] = scale_boxes(img_t.shape[2:], preds[:,:4], (h0,w0))
        for x1,y1,x2,y2,sc,lb in preds.cpu().numpy():
            ax.add_patch(patches.Rectangle((x1,y1),x2-x1,y2-y1,
                         lw=2, edgecolor="red", facecolor="none"))
            nm = names[int(lb)] if int(lb)<len(names) else str(int(lb))
            ax.text(x1, y1-2, f"{nm}:{sc:.2f}", color="red", fontsize=9,
                    backgroundcolor="black")
    ax.set_title(f"Detections (conf>{conf})")
    plt.tight_layout(); plt.savefig(out_path, dpi=110, bbox_inches="tight"); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default=None)
    ap.add_argument("--model-yaml", default="yolov8m.yaml")
    ap.add_argument("--nc", type=int, default=4)
    ap.add_argument("--names", default="holothurian,echinus,scallop,starfish")
    ap.add_argument("--stage1", type=int, default=0)
    ap.add_argument("--stage2", type=int, default=0)
    ap.add_argument("--dataset-name", default="DUO")
    ap.add_argument("--depth-cache", default="depth_cache")
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--n-images", type=int, default=4)
    ap.add_argument("--n-transmission", type=int, default=2)
    ap.add_argument("--out", default="viz_out")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    names = args.names.split(",")

    model = load_model(args.model_yaml, args.nc, args.stage1, args.stage2,
                       args.ckpt, device, args.n_transmission)
    depth_cache = DepthCache(args.depth_cache, args.dataset_name) \
        if (args.stage1 or args.stage2) else None

    imgs = sorted(glob.glob(os.path.join(args.img_dir, "*.jpg")))[:args.n_images]
    print(f"시각화 대상 이미지: {len(imgs)}개")

    # C. 학습된 파라미터 (이미지 무관, 1회)
    viz_params(model, os.path.join(args.out, "params.png"))
    print("  saved params.png (학습된 γ, α)")

    betas = []
    for k, img_path in enumerate(imgs):
        img_t, im0 = load_image_tensor(img_path, args.imgsz, device)
        stem = os.path.splitext(os.path.basename(img_path))[0]
        if depth_cache is not None:
            d = depth_cache.get(img_path, target_hw=img_t.shape[-2:]).unsqueeze(0).to(device)
            model.ppcm_prepare(img_t, d)
            betas.append(model._ppcm_beta[0].detach().cpu().numpy())

        if args.stage1:
            viz_stage1(model, img_t, os.path.join(args.out, f"{stem}_stage1.png"))
        if args.stage2:
            viz_stage2(model, img_t, os.path.join(args.out, f"{stem}_stage2.png"))
        viz_detection(model, img_path, depth_cache, device, args.imgsz, names,
                      os.path.join(args.out, f"{stem}_det.png"))
        print(f"  [{k+1}/{len(imgs)}] {stem}")

    if betas:
        viz_beta_dist(betas, os.path.join(args.out, "beta_distribution.png"))
        print("  saved beta_distribution.png")

    print(f"\n완료. 결과: {args.out}/")


if __name__ == "__main__":
    main()