"""
ppcm_viz.py — PPCM 새 파이프라인(SPEC) 시각화
==============================================
예전 viz 코드를 SPEC 에 맞게 재작성.

폐기된 것 (예전 → 삭제):
  - backscatter map      (Stage1 은 restoration 아님 → N1)
  - single weight map (exp(-βz)/mean 곱)  (Stage2 는 residual → P3)
  - "P3/P4/P5"           (YOLOv8 은 N3/N4/N5)

새로 그리는 것 (SPEC 이 실제 만드는 것):
  Stage1:  raw / corrected / 채널 gain / |diff|
  Stage2:  채널별 transmission (t_R,t_B) / γ·Conv modulation 크기
  물리해석: 학습된 γ_l, α, 예측 β_D 분포 (해석가능성 핵심)
  검출:    GT vs Pred (YOLOv8 출력 포맷)
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def _to_np(t):
    """(3,H,W)/(1,H,W)/(H,W) tensor → numpy, clip[0,1] if image."""
    t = t.detach().cpu().float()
    if t.dim() == 3 and t.shape[0] == 3:
        return np.clip(t.permute(1, 2, 0).numpy(), 0, 1)
    if t.dim() == 3 and t.shape[0] == 1:
        return t.squeeze(0).numpy()
    return t.numpy()


# =====================================================================
# Stage 1 — 채널 residual 재가중 (restoration 아님)
# =====================================================================
def viz_stage1(raw, corrected, beta, alpha, save_dir, epoch, batch_idx,
               max_images=2):
    """
    raw, corrected: [B,3,H,W] RGB [0,1]
    beta: [B,3]  예측된 β_D (채널 gain 계산용)
    alpha: float 스칼라 (학습된 Stage1 강도)
    columns: [Raw] [Corrected] [Channel Gain bar] [|Raw-Corr|]
    """
    os.makedirs(save_dir, exist_ok=True)
    n = min(raw.shape[0], max_images)
    fig, axes = plt.subplots(n, 4, figsize=(20, 5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    # 채널 gain = 1 + α(r_c - r̄),  r_c = exp(-β_c)
    r = torch.exp(-beta)                                   # [B,3]
    r_bar = r.mean(1, keepdim=True)
    gain = (1 + alpha * (r - r_bar)).detach().cpu().numpy()  # [B,3]

    for i in range(n):
        raw_np = _to_np(raw[i]); corr_np = _to_np(corrected[i])
        diff = np.abs(raw_np - corr_np)

        axes[i, 0].imshow(raw_np)
        axes[i, 0].set_title(f"Raw\nmean={raw_np.mean():.3f}", fontsize=10)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(corr_np)
        axes[i, 1].set_title(f"After Stage1 (residual)\nmean={corr_np.mean():.3f}",
                             fontsize=10)
        axes[i, 1].axis("off")

        # 채널 gain 막대 (R,G,B) — 1 기준 위/아래 = 강조/억제
        colors = ["red", "green", "blue"]
        axes[i, 2].bar(["R", "G", "B"], gain[i], color=colors, alpha=0.7)
        axes[i, 2].axhline(1.0, color="black", ls="--", lw=1)
        axes[i, 2].set_ylim(min(0.9, gain[i].min() - 0.02),
                            max(1.1, gain[i].max() + 0.02))
        axes[i, 2].set_title(f"Channel gain (1+α(r-r̄))\n"
                             f"β=[{beta[i,0]:.2f},{beta[i,1]:.2f},{beta[i,2]:.2f}]",
                             fontsize=10)

        im = axes[i, 3].imshow(diff.mean(2) if diff.ndim == 3 else diff, cmap="hot")
        axes[i, 3].set_title(f"|Raw-Corrected|\nmax={diff.max():.4f}", fontsize=10)
        axes[i, 3].axis("off")
        plt.colorbar(im, ax=axes[i, 3], fraction=0.046)

    plt.suptitle(f"PPCM Stage 1 (channel reliability, residual) — "
                 f"Ep{epoch} B{batch_idx}", fontsize=13)
    plt.tight_layout()
    path = os.path.join(save_dir, f"stage1_ep{epoch:02d}_b{batch_idx:04d}.png")
    plt.savefig(path, dpi=100, bbox_inches="tight"); plt.close()
    return path


# =====================================================================
# Stage 2 — 채널별 transmission + modulation 크기
# =====================================================================
def viz_stage2(transmission, feats_before, feats_after, gammas,
               save_dir, epoch, batch_idx):
    """
    transmission: [B,n_t,H,W]  (t_R, t_B) — 원해상
    feats_before/after: list of [B,C,Hl,Wl]  (N3,N4,N5 / N'3,N'4,N'5)
    gammas: list[float]  학습된 γ_l
    상단행: transmission t_R, t_B  (어느 파장이 어디서 신뢰 잃나)
    하단행: 스케일별 |N'-N| modulation 크기 (실제 feature 변화)
    """
    os.makedirs(save_dir, exist_ok=True)
    n_t = transmission.shape[1]
    n_scales = len(feats_before)
    ncols = max(n_t, n_scales)
    fig, axes = plt.subplots(2, ncols, figsize=(6 * ncols, 10))

    # ---- 상단: 채널별 transmission ----
    t_names = ["t_R (red transmission)", "t_B (blue transmission)"] if n_t == 2 \
        else ["t_R", "t_G", "t_B"]
    for j in range(ncols):
        ax = axes[0, j]
        if j < n_t:
            tmap = transmission[0, j].detach().cpu().numpy()
            im = ax.imshow(tmap, cmap="viridis", vmin=0, vmax=1)
            ax.set_title(f"{t_names[j]}\nmean={tmap.mean():.3f} "
                         f"(low=far/attenuated)", fontsize=10)
            plt.colorbar(im, ax=ax, fraction=0.046)
        ax.axis("off")

    # ---- 하단: 스케일별 modulation 크기 ----
    scale_names = ["N3 (small obj)", "N4 (medium)", "N5 (large)"]
    for j in range(ncols):
        ax = axes[1, j]
        if j < n_scales:
            mod = (feats_after[j] - feats_before[j]).abs().mean(1)[0]  # [Hl,Wl]
            mod = mod.detach().cpu().numpy()
            im = ax.imshow(mod, cmap="magma")
            g = gammas[j] if j < len(gammas) else float("nan")
            ax.set_title(f"{scale_names[j]}  |N'-N|\n"
                         f"γ={g:+.4f}, mean={mod.mean():.5f}", fontsize=10)
            plt.colorbar(im, ax=ax, fraction=0.046)
        ax.axis("off")

    plt.suptitle(f"PPCM Stage 2 — top: per-channel transmission (physics), "
                 f"bottom: actual feature modulation γ·Conv\nEp{epoch} B{batch_idx}",
                 fontsize=12)
    plt.tight_layout()
    path = os.path.join(save_dir, f"stage2_ep{epoch:02d}_b{batch_idx:04d}.png")
    plt.savefig(path, dpi=100, bbox_inches="tight"); plt.close()
    return path


# =====================================================================
# 물리 파라미터 추적 (해석가능성 핵심)
# =====================================================================
def viz_physics_params(history, save_dir):
    """
    history: dict of lists, 에폭별 기록
      'gamma': [[γ3,γ4,γ5], ...]   스케일별 γ 추이
      'alpha': [α, ...]             Stage1 강도 추이
      'beta_mean': [[βR,βG,βB], ...] 배치 평균 예측 β_D 추이
    학습된 γ_l 이 어느 스케일에서 열리는지 = "물리가 어느 스케일에 기여하나"
    """
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # γ_l 추이
    if "gamma" in history and history["gamma"]:
        g = np.array(history["gamma"])  # [E, n_scales]
        for l in range(g.shape[1]):
            axes[0].plot(g[:, l], marker="o", label=f"γ_N{l+3}")
        axes[0].axhline(0, color="gray", ls=":", lw=1)
        axes[0].set_title("Stage2 gate gamma_l per scale\n(which scale opens = physics contribution)")
        axes[0].set_xlabel("epoch"); axes[0].set_ylabel("γ")
        axes[0].legend()

    # α 추이
    if "alpha" in history and history["alpha"]:
        axes[1].plot(history["alpha"], marker="s", color="purple")
        axes[1].set_title("Stage1 strength α")
        axes[1].set_xlabel("epoch"); axes[1].set_ylabel("α")

    # β_D 예측 추이
    if "beta_mean" in history and history["beta_mean"]:
        b = np.array(history["beta_mean"])  # [E,3]
        for c, name, col in zip(range(3), ["βR", "βG", "βB"],
                                ["red", "green", "blue"]):
            axes[2].plot(b[:, c], marker="^", color=col, label=name)
        axes[2].set_title("Predicted β_D (batch mean)\n"
                          "(does water-type prediction converge to physical range)")
        axes[2].set_xlabel("epoch"); axes[2].set_ylabel("β")
        axes[2].legend()

    plt.suptitle("PPCM learned physics parameters (interpretability)", fontsize=13)
    plt.tight_layout()
    path = os.path.join(save_dir, "physics_params.png")
    plt.savefig(path, dpi=100, bbox_inches="tight"); plt.close()
    return path


def viz_beta_distribution(beta_all, save_dir, tag="train"):
    """
    beta_all: [N,3]  전체 데이터셋의 예측 β_D
    앵커(Solonenko&Mobley 6점) 위에 예측 분포를 겹쳐 그림.
    → check predicted beta near physical anchors (해석가능성).
    """
    os.makedirs(save_dir, exist_ok=True)
    from ppcm_modules import PhysInterp
    anchors = PhysInterp().anchors.numpy()  # [6,3]
    names = ["I", "II", "III", "1C", "5C", "9C"]

    beta_all = np.asarray(beta_all)
    fig, ax = plt.subplots(figsize=(8, 6))
    # βR vs βB 평면
    ax.scatter(beta_all[:, 2], beta_all[:, 0], s=8, alpha=0.4,
               color="steelblue", label="predicted β")
    ax.scatter(anchors[:, 2], anchors[:, 0], s=200, marker="X",
               color="crimson", edgecolor="black", zorder=5,
               label="S&M anchors")
    for i, nm in enumerate(names):
        ax.annotate(nm, (anchors[i, 2], anchors[i, 0]),
                    fontsize=11, fontweight="bold",
                    xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("β_B (turbidity axis)")
    ax.set_ylabel("β_R")
    ax.set_title(f"Predicted β_D vs physical anchors ({tag})\n"
                 "(predictions near anchors = physically valid water-type)")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(save_dir, f"beta_dist_{tag}.png")
    plt.savefig(path, dpi=100, bbox_inches="tight"); plt.close()
    return path


# =====================================================================
# 검출 결과 (YOLOv8 출력 포맷)
# =====================================================================
def viz_detections(images, preds, targets, categories, save_dir,
                   epoch, batch_idx, max_images=2, score_thresh=0.3):
    """
    images: [B,3,H,W] [0,1]
    preds:  list of dict {boxes:[N,4] xyxy, scores:[N], labels:[N]}  (YOLOv8)
    targets: list of dict {boxes, labels}  (GT)
    categories: {id:name}
    GT=lime, Pred=red
    """
    os.makedirs(save_dir, exist_ok=True)
    n = min(images.shape[0], max_images)
    fig, axes = plt.subplots(1, n, figsize=(12 * n, 8))
    if n == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.imshow(_to_np(images[i])); ax.axis("off")
        if targets and i < len(targets):
            for box, lb in zip(targets[i]["boxes"].cpu().numpy(),
                               targets[i]["labels"].cpu().numpy()):
                x1, y1, x2, y2 = box
                ax.add_patch(patches.Rectangle((x1, y1), x2-x1, y2-y1,
                             linewidth=2, edgecolor="lime", facecolor="none"))
                ax.text(x1, y1-2, f"GT:{categories.get(int(lb), lb)}",
                        color="lime", fontsize=8, backgroundcolor="black")
        if preds and i < len(preds):
            for box, sc, lb in zip(preds[i]["boxes"].cpu().numpy(),
                                   preds[i]["scores"].cpu().numpy(),
                                   preds[i]["labels"].cpu().numpy()):
                if sc < score_thresh:
                    continue
                x1, y1, x2, y2 = box
                ax.add_patch(patches.Rectangle((x1, y1), x2-x1, y2-y1,
                             linewidth=2, edgecolor="red", facecolor="none"))
                ax.text(x1, y2+2, f"{categories.get(int(lb), lb)}:{sc:.2f}",
                        color="red", fontsize=8, backgroundcolor="black")
        ax.set_title(f"Image {i} | GT=lime Pred=red (>{score_thresh:.0%})")

    plt.suptitle(f"Detections — Ep{epoch} B{batch_idx}", fontsize=12)
    plt.tight_layout()
    path = os.path.join(save_dir, f"det_ep{epoch:02d}_b{batch_idx:04d}.png")
    plt.savefig(path, dpi=100, bbox_inches="tight"); plt.close()
    return path


def viz_training_curve(losses, save_dir, extra=None):
    """losses: list[float]. extra: optional dict{name:list} for multi-curve."""
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(losses, marker="o", lw=2, label="total")
    if extra:
        for name, vals in extra.items():
            ax.plot(vals, marker=".", label=name, alpha=0.7)
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.set_title("Training Loss"); ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout()
    path = os.path.join(save_dir, "training_curve.png")
    plt.savefig(path, dpi=100); plt.close()
    return path