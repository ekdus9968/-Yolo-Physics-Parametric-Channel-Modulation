"""
visualize_results.py -- PPCM result/analysis visualization suite (5 types)
==========================================================================
1. save_config_comparison    : mAP50/small/medium/large 4 sections, 4 configs
2. save_stage1_visualization : raw / corrected / channel-residual / |diff| 4 cols
3. save_stage2_visualization : P3/P4/P5 spatial weight map (green=trust, red=distrust)
4. save_detection_visualization: GT box(green) vs predicted box(red)
5. save_training_curve       : loss_log.csv -> loss curve

Usage:
  # config comparison (needs ablation json)
  python visualize_results.py --mode compare --ablation-json ablation_pathB_duo.json --out viz_out

  # stage1/stage2/detection maps (needs checkpoint)
  python visualize_results.py --mode maps \
      --ckpt runs/ppcm_pathB/DUO_full/full_best.pt --stage1 1 --stage2 1 \
      --img-dir data/DUO/test/images --label-dir data/DUO/test/labels \
      --dataset-name DUO --depth-cache depth_cache --n-images 4 --out viz_out

  # training curve
  python visualize_results.py --mode curve \
      --loss-csv runs/ppcm_pathB/DUO_stage2/loss_log.csv --out viz_out
"""
"""
PPCM Visualization — clean, side-by-side comparison layout.

All-in-one: baseline / stage1 / stage2 / full on same canvas.
"""
import os, csv, glob, json, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import TwoSlopeNorm
import torch
import torch.nn.functional as F

# ── helpers ──────────────────────────────────────────────────────────
def _to_img(t):
    return np.clip(t.detach().cpu().float().permute(1,2,0).numpy(), 0, 1)

CONFIG_COLORS = {
    "baseline": "#73726c",
    "stage1":   "#1baf7a",
    "stage2":   "#2a78d6",
    "full":     "#e34948",
}
CONFIG_LABELS = {
    "baseline": "Baseline",
    "stage1":   "+ Stage 1",
    "stage2":   "+ Stage 2",
    "full":     "Full PPCM",
}


# ── 1. Config comparison bar chart ───────────────────────────────────
def save_config_comparison(ablation_json, out_path):
    with open(ablation_json) as f:
        res = json.load(f)

    configs  = [c for c in ["baseline","stage1","stage2","full"] if c in res]
    metrics  = [("mAP50","mAP@50"), ("AP_small","AP small"),
                ("AP_medium","AP medium"), ("AP_large","AP large")]

    fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 5))
    for ax, (key, title) in zip(axes, metrics):
        vals = [res[c].get(key, 0) for c in configs]
        base = res.get("baseline",{}).get(key, None)
        bars = ax.bar([CONFIG_LABELS[c] for c in configs], vals,
                      color=[CONFIG_COLORS[c] for c in configs], alpha=0.85, width=0.5)
        if base is not None:
            ax.axhline(base, color="#aaa", ls="--", lw=1)
        for b, c, v in zip(bars, configs, vals):
            d    = v - base if (base is not None and c != "baseline") else None
            text = f"{v:.1f}" if d is None else f"{v:.1f}\n({d:+.1f})"
            ax.text(b.get_x()+b.get_width()/2, v+0.2, text,
                    ha="center", va="bottom", fontsize=9)
        lo = min(vals+([base] if base else []))-3
        hi = max(vals+([base] if base else []))+5
        ax.set_ylim(lo, hi); ax.set_title(title, fontsize=12, fontweight="bold")
        ax.tick_params(axis="x", labelsize=9); ax.grid(axis="y", alpha=0.25)
    fig.suptitle("PPCM Ablation — 4-config comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close()
    print(f"  saved {out_path}")


# ── 2. Stage 1 — 4 panels per image ──────────────────────────────────
def save_stage1_visualization(raw_t, corrected_t, beta, alpha, out_path):
    """
    raw_t, corrected_t: (3,H,W) tensors [0,1]
    beta: (3,) R,G,B attenuation
    """
    raw  = _to_img(raw_t)
    corr = _to_img(corrected_t)
    res  = corr - raw              # signed residual

    fig, ax = plt.subplots(1, 4, figsize=(20, 5))

    ax[0].imshow(raw)
    ax[0].set_title(f"Raw  mean={raw.mean():.3f}", fontsize=11)

    ax[1].imshow(corr)
    ax[1].set_title(f"Corrected  mean={corr.mean():.3f}", fontsize=11)

    # signed residual — green=boosted, red=suppressed
    rm   = res.mean(2)
    vmax = max(abs(rm.min()), abs(rm.max()), 1e-6)
    im2  = ax[2].imshow(rm, cmap="RdYlGn",
                        norm=TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax))
    ax[2].set_title("Residual (green=trust, red=suppress)", fontsize=11)
    plt.colorbar(im2, ax=ax[2], fraction=0.046)

    # magnitude
    im3 = ax[3].imshow(np.abs(rm), cmap="hot")
    ax[3].set_title(f"|residual|  max={np.abs(rm).max():.4f}", fontsize=11)
    plt.colorbar(im3, ax=ax[3], fraction=0.046)

    for a in ax: a.axis("off")
    b = beta if isinstance(beta, (list,tuple)) else beta.tolist()
    fig.suptitle(f"Stage 1 — channel reliability reweighting  "
                 f"α={alpha:.3f}  β=[{b[0]:.3f}, {b[1]:.3f}, {b[2]:.3f}]",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight"); plt.close()


# ── 3. Stage 2 — spatial weight maps P3/P4/P5 ────────────────────────
def save_stage2_visualization(weight_maps, img_t, out_path):
    """
    weight_maps: dict or list of (1,1,H,W) or (H,W) tensors
                 key/index order: P3, P4, P5
    img_t: (3,H,W) input image
    """
    if isinstance(weight_maps, dict):
        items = list(weight_maps.items())
    else:
        items = [(f"scale{i}", w) for i, w in enumerate(weight_maps)]

    n   = len(items)
    fig, ax = plt.subplots(1, n+1, figsize=(6*(n+1), 5))

    ax[0].imshow(_to_img(img_t))
    ax[0].set_title("Input", fontsize=11); ax[0].axis("off")

    scale_names = ["P3 (small)", "P4 (medium)", "P5 (large)"]
    for i, (key, w) in enumerate(items):
        if isinstance(w, torch.Tensor):
            w = w.detach().cpu().squeeze().numpy()
        vmax = max(abs(w.min()), abs(w.max()), 1e-6)
        im = ax[i+1].imshow(w, cmap="RdYlGn",
                            norm=TwoSlopeNorm(vcenter=1.0, vmin=max(0,1-vmax),
                                              vmax=1+vmax))
        nm = scale_names[i] if i < len(scale_names) else key
        ax[i+1].set_title(f"{nm}\n"
                          f"mean={w.mean():.3f}  std={w.std():.3f}", fontsize=11)
        ax[i+1].axis("off"); plt.colorbar(im, ax=ax[i+1], fraction=0.046)

    fig.suptitle("Stage 2 — spatial reliability weight  "
                 "(green=shallow/trust, red=deep/suppress)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight"); plt.close()


# ── 4. Detection — 4 configs side-by-side ────────────────────────────
def save_detection_comparison(img_np, results_dict, gt_boxes, gt_labels,
                               names, out_path, conf=0.25):
    """
    img_np: (H,W,3) float [0,1] or uint8
    results_dict: {"baseline": [(x1,y1,x2,y2,score,cls), ...], "stage1": ..., ...}
    gt_boxes: [(x1,y1,x2,y2), ...]  pixel coords
    gt_labels: [int, ...]
    """
    configs = [c for c in ["baseline","stage1","stage2","full"] if c in results_dict]
    n = len(configs) + 1   # +1 for GT

    fig, axes = plt.subplots(1, n, figsize=(6*n, 5))

    def _draw(ax, title, boxes, labels=None, scores=None, color="red", dashed=False):
        ax.imshow(img_np); ax.axis("off"); ax.set_title(title, fontsize=11)
        for i, (x1,y1,x2,y2) in enumerate(boxes):
            ls  = "--" if dashed else "-"
            ax.add_patch(patches.Rectangle((x1,y1), x2-x1, y2-y1,
                         lw=2, edgecolor=color, facecolor="none", ls=ls))
            if labels is not None:
                lbl = names[int(labels[i])] if names else str(int(labels[i]))
                txt = f"{lbl}" if scores is None else f"{lbl} {scores[i]:.2f}"
                ax.text(x1, max(y1-3,0), txt, fontsize=7, color=color,
                        bbox=dict(facecolor="black",alpha=0.45,pad=1,lw=0))

    # GT
    _draw(axes[0], f"Ground Truth ({len(gt_boxes)})",
          gt_boxes, gt_labels, color="#27ae60")

    # each config
    for i, c in enumerate(configs):
        dets   = [(x1,y1,x2,y2,sc,lb) for x1,y1,x2,y2,sc,lb
                  in results_dict[c] if sc >= conf]
        boxes  = [(x1,y1,x2,y2) for x1,y1,x2,y2,sc,lb in dets]
        scores = [sc for x1,y1,x2,y2,sc,lb in dets]
        labels = [lb for x1,y1,x2,y2,sc,lb in dets]
        _draw(axes[i+1],
              f"{CONFIG_LABELS[c]} ({len(boxes)} det)",
              boxes, labels, scores, CONFIG_COLORS[c], dashed=True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight"); plt.close()


# ── 5. Training curve ─────────────────────────────────────────────────
def save_training_curve(loss_source, out_path):
    """
    loss_source: path to loss_log.csv  OR  list of float (epoch avg losses)
    """
    if isinstance(loss_source, str) and os.path.exists(loss_source):
        steps, losses, boxes, clss, dfls = [], [], [], [], []
        with open(loss_source) as f:
            for i, row in enumerate(csv.DictReader(f)):
                steps.append(i)
                losses.append(float(row.get("loss","nan")))
                boxes.append(float(row.get("box","nan")))
                clss.append(float(row.get("cls","nan")))
                dfls.append(float(row.get("dfl","nan")))
        has_components = True
    else:
        # plain list of per-epoch avg losses
        losses = list(loss_source)
        steps  = list(range(len(losses)))
        has_components = False

    ncols = 2 if has_components else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7*ncols, 4))
    ax0 = axes[0] if has_components else axes

    ax0.plot(steps, losses, color="#2a78d6", lw=1.5)
    ax0.set_title("Total loss"); ax0.set_xlabel("step / epoch")
    ax0.set_ylabel("loss"); ax0.grid(alpha=0.25)

    if has_components:
        axes[1].plot(steps, boxes, label="box", color="#1baf7a", lw=1.2)
        axes[1].plot(steps, clss,  label="cls", color="#e24b4a", lw=1.2)
        axes[1].plot(steps, dfls,  label="dfl", color="#eda100", lw=1.2)
        axes[1].set_title("Loss components"); axes[1].set_xlabel("step")
        axes[1].legend(); axes[1].grid(alpha=0.25)

    plt.suptitle("Training curve", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close()
    print(f"  saved {out_path}")


# ── 6. Per-class AP heatmap (new — easy to see where physics helps) ───
def save_per_class_heatmap(ablation_json, class_names, out_path):
    """
    Shows per-class AP for each config as a colour matrix.
    Quickly reveals which classes benefit from physics correction.
    """
    with open(ablation_json) as f:
        res = json.load(f)

    configs = [c for c in ["baseline","stage1","stage2","full"] if c in res]
    # expect res[config]["per_class"] = {classname: AP}
    if "per_class" not in res.get("baseline", {}):
        print("[skip] per_class not in ablation json"); return

    mat = np.array([[res[c]["per_class"].get(n, 0) for n in class_names]
                    for c in configs])          # (n_config, n_class)
    delta = mat - mat[0:1, :]                   # delta vs baseline

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(len(class_names)*2+3, 5))

    # absolute AP
    im1 = ax1.imshow(mat, cmap="YlGn", vmin=0, vmax=100, aspect="auto")
    ax1.set_xticks(range(len(class_names))); ax1.set_xticklabels(class_names, rotation=30)
    ax1.set_yticks(range(len(configs)));    ax1.set_yticklabels([CONFIG_LABELS[c] for c in configs])
    ax1.set_title("Per-class AP", fontweight="bold")
    plt.colorbar(im1, ax=ax1)
    for i in range(len(configs)):
        for j in range(len(class_names)):
            ax1.text(j, i, f"{mat[i,j]:.1f}", ha="center", va="center", fontsize=9)

    # delta vs baseline
    vext = max(abs(delta[1:].min()), abs(delta[1:].max()), 1e-3)
    im2  = ax2.imshow(delta, cmap="RdYlGn",
                      norm=TwoSlopeNorm(vcenter=0, vmin=-vext, vmax=vext), aspect="auto")
    ax2.set_xticks(range(len(class_names))); ax2.set_xticklabels(class_names, rotation=30)
    ax2.set_yticks(range(len(configs)));    ax2.set_yticklabels([CONFIG_LABELS[c] for c in configs])
    ax2.set_title("Δ AP vs baseline", fontweight="bold")
    plt.colorbar(im2, ax=ax2)
    for i in range(len(configs)):
        for j in range(len(class_names)):
            ax2.text(j, i, f"{delta[i,j]:+.1f}", ha="center", va="center", fontsize=9)

    plt.suptitle("Per-class AP — where does physics correction help?",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close()
    print(f"  saved {out_path}")


# ── main ──────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["compare", "maps", "curve", "per-class", "detection", "all"])
    ap.add_argument("--out", default="viz_out")

    # compare / per-class 공통
    ap.add_argument("--ablation-json", default="ablation_pathB_duo.json")

    # per-class 전용
    ap.add_argument("--class-names",
                    default="holothurian,echinus,scallop,starfish",
                    help="쉼표로 구분")

    # maps / detection 공통
    ap.add_argument("--ckpt", default=None,
                    help="maps/detection: 단일 체크포인트")
    ap.add_argument("--ckpt-dir", default=None,
                    help="detection 4구성 비교용: 각 구성별 체크포인트 폴더 루트")
    ap.add_argument("--ckpt-pattern", default="{cfg}_best.pt",
                    help="detection용: ckpt-dir 안에서 파일 찾을 패턴")
    ap.add_argument("--model-yaml", default="yolov8m.yaml")
    ap.add_argument("--nc", type=int, default=4)
    ap.add_argument("--stage1", type=int, default=0)
    ap.add_argument("--stage2", type=int, default=0)
    ap.add_argument("--dataset-name", default="DUO")
    ap.add_argument("--depth-cache", default="depth_cache")
    ap.add_argument("--img-dir", default=None)
    ap.add_argument("--label-dir", default=None)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--n-images", type=int, default=4)
    ap.add_argument("--n-transmission", type=int, default=2)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)

    # curve 전용
    ap.add_argument("--loss-csv", default=None)

    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    class_names = [n.strip() for n in args.class_names.split(",")]

    # ── 1) config 비교 막대 ────────────────────────────────────────────
    if args.mode in ("compare", "all"):
        if os.path.exists(args.ablation_json):
            save_config_comparison(
                args.ablation_json,
                os.path.join(args.out, "config_comparison.png"))
        else:
            print(f"[skip] compare: {args.ablation_json} 없음")

    # ── 2) per-class AP heatmap ───────────────────────────────────────
    if args.mode in ("per-class", "all"):
        if os.path.exists(args.ablation_json):
            save_per_class_heatmap(
                args.ablation_json, class_names,
                os.path.join(args.out, "per_class_heatmap.png"))
        else:
            print(f"[skip] per-class: {args.ablation_json} 없음")

    # ── 3) training curve ─────────────────────────────────────────────
    if args.mode in ("curve", "all"):
        if args.loss_csv and os.path.exists(args.loss_csv):
            save_training_curve(
                args.loss_csv,
                os.path.join(args.out, "training_curve.png"))
        else:
            print(f"[skip] curve: --loss-csv 없음")

    # ── 4) stage1 / stage2 맵 ─────────────────────────────────────────
    if args.mode in ("maps", "all"):
        if not args.ckpt:
            print("[skip] maps: --ckpt 필요")
        elif not args.img_dir:
            print("[skip] maps: --img-dir 필요")
        else:
            model = load_model(args.model_yaml, args.nc, args.stage1, args.stage2,
                               args.ckpt, device, args.n_transmission)
            depth_cache = DepthCache(args.depth_cache, args.dataset_name) \
                if (args.stage1 or args.stage2) else None
            imgs = sorted(glob.glob(os.path.join(args.img_dir, "*.jpg")))[:args.n_images]
            print(f"maps: {len(imgs)}장 처리")
            for k, img_path in enumerate(imgs):
                stem = os.path.splitext(os.path.basename(img_path))[0]
                img_t, im0, _ = load_image_tensor(img_path, args.imgsz, device)
                if depth_cache is not None:
                    d = depth_cache.get(img_path, target_hw=img_t.shape[-2:]) \
                                   .unsqueeze(0).to(device)
                    model.ppcm_prepare(img_t, d)

                if args.stage1:
                    beta = model._ppcm_beta[0].detach().cpu().numpy()
                    alpha = float(model.ppcm.stage1.alpha)
                    corrected = model.ppcm.stage1(img_t, model._ppcm_beta)
                    save_stage1_visualization(
                        img_t[0], corrected[0], beta, alpha,
                        os.path.join(args.out, f"{stem}_stage1.png"))

                if args.stage2:
                    # stage2 weight maps는 코드에서 별도 계산 필요 (실제 구현은
                    # ppcm_yolo_model 내부에서 나오는 값을 넘겨줘야 함)
                    # 여기선 placeholder — 실제 hook 방식이 정해지면 채워야 함
                    print(f"  [note] stage2 weight_maps 추출은 별도 구현 필요")

                print(f"  [{k+1}/{len(imgs)}] {stem}")

    # ── 5) detection 4구성 비교 ───────────────────────────────────────
    if args.mode in ("detection", "all"):
        if not args.ckpt_dir:
            print("[skip] detection: --ckpt-dir 필요")
        elif not args.img_dir:
            print("[skip] detection: --img-dir 필요")
        else:
            from ultralytics.utils.nms import non_max_suppression
            from ultralytics.utils.ops import scale_boxes
            import cv2

            configs = [("baseline",0,0), ("stage1",1,0),
                       ("stage2",0,1), ("full",1,1)]
            # 4개 체크포인트 존재하는 것만
            available = []
            models = {}
            for name, s1, s2 in configs:
                ckpt = os.path.join(args.ckpt_dir, args.ckpt_pattern.format(cfg=name))
                if os.path.exists(ckpt):
                    models[name] = (load_model(args.model_yaml, args.nc, s1, s2,
                                               ckpt, device, args.n_transmission),
                                    s1, s2)
                    available.append(name)
                else:
                    print(f"  [skip] {name}: {ckpt} 없음")

            if not models:
                print("[skip] detection: 사용 가능한 체크포인트 없음")
            else:
                depth_cache = DepthCache(args.depth_cache, args.dataset_name)
                imgs = sorted(glob.glob(os.path.join(args.img_dir, "*.jpg")))[:args.n_images]
                print(f"detection: {len(imgs)}장 × {len(models)}구성")

                for k, img_path in enumerate(imgs):
                    stem = os.path.splitext(os.path.basename(img_path))[0]
                    img_t, im0, _ = load_image_tensor(img_path, args.imgsz, device)
                    h0, w0 = im0.shape[:2]

                    # GT 로드
                    gt_boxes, gt_labels = [], []
                    if args.label_dir:
                        lp = os.path.join(args.label_dir, f"{stem}.txt")
                        if os.path.exists(lp):
                            with open(lp) as f:
                                for line in f:
                                    p = line.split()
                                    if len(p) < 5: continue
                                    cls, cx, cy, bw, bh = int(p[0]), *map(float, p[1:5])
                                    x1 = (cx-bw/2)*w0; y1 = (cy-bh/2)*h0
                                    x2 = (cx+bw/2)*w0; y2 = (cy+bh/2)*h0
                                    gt_boxes.append((x1,y1,x2,y2))
                                    gt_labels.append(cls)

                    # 각 구성별 예측
                    results_dict = {}
                    for name in available:
                        model, s1, s2 = models[name]
                        if s1 or s2:
                            d = depth_cache.get(img_path, target_hw=img_t.shape[-2:]) \
                                           .unsqueeze(0).to(device)
                            model.ppcm_prepare(img_t, d)
                        with torch.no_grad():
                            out = model(img_t)
                        preds = out[0] if isinstance(out, (list,tuple)) else out
                        preds = non_max_suppression(preds, args.conf, args.iou,
                                                    max_det=300)[0]
                        dets = []
                        if preds is not None and len(preds):
                            preds[:, :4] = scale_boxes(img_t.shape[2:], preds[:, :4],
                                                       (h0, w0))
                            for x1,y1,x2,y2,sc,lb in preds.cpu().numpy():
                                dets.append((x1,y1,x2,y2, float(sc), int(lb)))
                        results_dict[name] = dets

                    img_np = im0[:, :, ::-1] / 255.0    # BGR→RGB, [0,1]
                    save_detection_comparison(
                        img_np, results_dict, gt_boxes, gt_labels,
                        class_names,
                        os.path.join(args.out, f"{stem}_detection.png"),
                        conf=args.conf)
                    print(f"  [{k+1}/{len(imgs)}] {stem}")

    print(f"\n완료. 결과: {args.out}/")


if __name__ == "__main__":
    main()