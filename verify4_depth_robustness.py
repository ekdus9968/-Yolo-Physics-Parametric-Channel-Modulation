#!/usr/bin/env python3
"""
검증 4 — 채널비율의 Depth-Robustness (pseudo-label 설계 정당화)
==============================================================

배경
----
supervision 설계에서 water-type pseudo-label 로 채널비율 log(I_R/I_B) 를 쓰기로 함.
근거: "채널비율은 depth 영향을 상쇄하고 water type 스펙트럼만 반영한다".
그러나 검증3 에서 '이미지 전체' 채널비율은 depth 와 강하게 결합됨을 확인.
→ 모순 아님. 관건은 '어느 영역'의 채널비율을 쓰느냐:
    - 전체 영역:  depth 의존 큼   (water type 학습에 부적합)
    - 근거리 영역: depth 의존 작음  (water type 만 남음, pseudo-label 로 적합)

검증 4 가 보여야 할 것
----------------------
"근거리 한정 채널비율"이 "전체 채널비율"보다 depth 와의 상관이 유의하게 낮다.
그 gap 이 클수록 → 근거리 제한이 depth 를 성공적으로 상쇄 → pseudo-label 방어됨.

무엇을 뒷받침하나
-----------------
지난 논의의 우려("red/채널비율이 depth 와 교락")를 '근거리 제한'으로 푼다는
주장을 데이터로 입증. Stage1 의 water-type 예측기 g_φ 가 depth 가 아니라
수형을 학습하도록 만드는 근거.

측정 방법
---------
각 이미지에서:
  1) depth proxy z 계산 (검증3 과 동일 함수)
  2) 전체 픽셀의 (z, logRB) 상관 ρ_all
  3) 근거리(밝기 상위 p%) 픽셀의 (z_near, logRB_near) 상관 ρ_near
  4) |ρ_all| - |ρ_near| = depth 상쇄 효과 (클수록 좋음)

또한 이미지 '한 장당 하나의 스칼라' pseudo-label 을 만들 때
  near-field 평균 logRB 가 depth 분포에 얼마나 둔감한지도 확인:
  이미지들의 (평균 depth, near-field logRB) 산점 → 상관 낮아야 함.

사용
----
python verify4_depth_robustness.py --config datasets.yaml
python verify4_depth_robustness.py --config datasets.yaml --depth-dir data/depth_maps
python verify4_depth_robustness.py --demo
"""

import argparse
import glob
import os
import sys
import numpy as np
import cv2
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 검증3의 depth_proxy 재사용 (동일 구현)
def depth_proxy(img_bgr):
    # G 채널 전용 (검증3과 동일). R,B 배제로 log(R/B)와의 정의상 순환 제거.
    img = img_bgr.astype(np.float32) / 255.0
    G = img[..., 1]
    blur = cv2.GaussianBlur(G, (0, 0), sigmaX=4)
    local_contrast = cv2.GaussianBlur((G - blur) ** 2, (0, 0), sigmaX=4)
    local_contrast = np.sqrt(np.maximum(local_contrast, 0))
    def norm(x):
        lo, hi = np.percentile(x, 2), np.percentile(x, 98)
        return np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)
    return 1.0 - norm(local_contrast)

def load_depth_map(depth_path, target_hw):
    if depth_path.endswith(".npy"):
        z = np.load(depth_path).astype(np.float32)
    else:
        z = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if z is None:
            return None
        z = z.astype(np.float32)
        if z.ndim == 3:
            z = z.mean(axis=2)
    z = (z - z.min()) / (z.max() - z.min() + 1e-6)
    return cv2.resize(z, (target_hw[1], target_hw[0]))


def robustness_for_image(img_bgr, z=None, near_pct=80, eps=1e-4):
    img = img_bgr.astype(np.float32) / 255.0
    B, G, R = img[..., 0], img[..., 1], img[..., 2]
    logRB = np.log((R + eps) / (B + eps))
    lum = (R + G + B) / 3.0
    if z is None:
        z = depth_proxy(img_bgr)

    zf, lf, lumf = z.flatten(), logRB.flatten(), lum.flatten()
    valid = (lf > np.percentile(lf, 1)) & (lf < np.percentile(lf, 99))
    zf, lf, lumf = zf[valid], lf[valid], lumf[valid]
    if len(zf) < 200:
        return None

    # 전체 상관
    rho_all, _ = spearmanr(zf, lf)

    # 근거리(밝기 상위) 상관
    thr = np.percentile(lumf, near_pct)
    near = lumf >= thr
    if near.sum() < 100:
        return None
    rho_near, _ = spearmanr(zf[near], lf[near])

    return {
        "rho_all": float(rho_all),
        "rho_near": float(rho_near),
        "depth_cancel": float(abs(rho_all) - abs(rho_near)),  # >0 이면 근거리가 depth 상쇄
        "near_logRB": float(lf[near].mean()),   # 이미지당 pseudo-label 후보
        "mean_depth": float(zf.mean()),
    }


def process_dataset(name, paths, depth_dir=None, max_images=300):
    if max_images:
        paths = paths[:max_images]
    rows = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        z = None
        if depth_dir:
            stem = os.path.splitext(os.path.basename(p))[0]
            for ext in (".npy", ".png"):
                dp = os.path.join(depth_dir, stem + ext)
                if os.path.exists(dp):
                    z = load_depth_map(dp, img.shape[:2]); break
        res = robustness_for_image(img, z=z)
        if res:
            res.update({"dataset": name, "path": p})
            rows.append(res)
    df = pd.DataFrame(rows)
    if len(df):
        print(f"  [{name}] {len(df)} imgs | ρ_all={df.rho_all.mean():+.3f} "
              f"ρ_near={df.rho_near.mean():+.3f} cancel={df.depth_cancel.mean():+.3f}")
    return df


def plot_robustness(all_df, out_dir):
    datasets = sorted(all_df["dataset"].unique())
    cmap = plt.get_cmap("tab10")

    # (1) ρ_all vs ρ_near 막대 비교
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(datasets))
    w = 0.35
    all_means = [all_df[all_df.dataset == d]["rho_all"].abs().mean() for d in datasets]
    near_means = [all_df[all_df.dataset == d]["rho_near"].abs().mean() for d in datasets]
    ax.bar(x - w/2, all_means, w, label="|ρ| whole image", color="#c0392b", alpha=0.8)
    ax.bar(x + w/2, near_means, w, label="|ρ| near-field only", color="#27ae60", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(datasets, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("|Spearman ρ|  (depth vs log R/B)")
    ax.set_title("Depth-robustness of channel ratio\n"
                 "near-field bars LOWER than whole-image = depth successfully cancelled")
    ax.legend()
    fig.tight_layout()
    p1 = os.path.join(out_dir, "fig_depth_robustness_bars.png")
    fig.savefig(p1, dpi=140); plt.close(fig); print(f"  saved {p1}")

    # (2) 이미지당 (mean_depth, near_logRB) 산점 — pseudo-label 이 depth 에 둔감한가
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, d in enumerate(datasets):
        sub = all_df[all_df.dataset == d]
        ax.scatter(sub["mean_depth"], sub["near_logRB"], s=12, alpha=0.5,
                   color=cmap(i % 10), label=d)
    ax.set_xlabel("mean depth of image (proxy)")
    ax.set_ylabel("near-field log(R/B)  (pseudo-label candidate)")
    ax.set_title("Pseudo-label vs image depth\n"
                 "flat cloud (no trend) = pseudo-label encodes water type, not depth")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p2 = os.path.join(out_dir, "fig_pseudolabel_vs_depth.png")
    fig.savefig(p2, dpi=140); plt.close(fig); print(f"  saved {p2}")


def make_demo_image(seed=0, near_pct=80):
    rng = np.random.default_rng(seed)
    H, W = 120, 160
    J = rng.uniform(0.4, 0.9, size=(H, W, 3)).astype(np.float32)
    z = np.linspace(0.1, 1.0, H)[:, None] * np.ones((1, W))
    z = (z + rng.normal(0, 0.05, (H, W))).clip(0, 1).astype(np.float32)
    out = np.zeros((H, W, 3), np.float32)
    betas = [0.2, 0.5, 1.2]; Binf = [0.15, 0.12, 0.08]
    for c in range(3):
        t = np.exp(-betas[c] * z * 3)
        out[..., c] = J[..., c] * t + Binf[c] * (1 - t)
    return (out.clip(0, 1) * 255).astype(np.uint8)


def run_demo():
    print("[DEMO] near-field should cancel depth coupling...")
    rows = []
    for i in range(80):
        img = make_demo_image(seed=i)
        res = robustness_for_image(img)
        if res:
            res.update({"dataset": "demo_coupled", "path": f"d_{i}"})
            rows.append(res)
    df = pd.DataFrame(rows)
    print(f"  ρ_all={df.rho_all.mean():+.3f}  ρ_near={df.rho_near.mean():+.3f}  "
          f"cancel={df.depth_cancel.mean():+.3f}")
    return df


def load_config(cfg_path):
    # 인코딩 견고화: UTF-8(BOM 포함) 우선, 실패 시 cp949/latin-1 폴백
    text = None
    for enc in ("utf-8-sig", "utf-8", "cp949", "latin-1"):
        try:
            with open(cfg_path, encoding=enc) as f:
                text = f.read()
            break
        except (UnicodeDecodeError, LookupError):
            continue
    if text is None:
        raise RuntimeError(f"cannot decode {cfg_path}")
    m = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        k, v = line.split(":", 1)
        m[k.strip()] = v.split("#")[0].strip()
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--depth-dir")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--max-images", type=int, default=300)
    ap.add_argument("--outdir", default="verify4_out")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.demo or not args.config:
        all_df = run_demo()
    else:
        cfg = load_config(args.config)
        frames = []
        for name, pattern in cfg.items():
            paths = sorted(glob.glob(pattern))
            if not paths:
                print(f"  [WARN] no images: {name}", file=sys.stderr); continue
            frames.append(process_dataset(name, paths, args.depth_dir, args.max_images))
        if not frames:
            print("No data.", file=sys.stderr); sys.exit(1)
        all_df = pd.concat(frames, ignore_index=True)

    all_df.to_csv(os.path.join(args.outdir, "robustness_per_image.csv"), index=False)

    print("\n=== Depth-robustness summary ===")
    summ = all_df.groupby("dataset")[["rho_all", "rho_near", "depth_cancel"]].mean()
    print(summ.round(3).to_string())
    print("\n해석: depth_cancel = |ρ_all| - |ρ_near| > 0 이면 근거리 제한이 depth 를 상쇄.")
    print("      → near-field 채널비율을 pseudo-label 로 쓰는 것이 정당화됨.")
    summ.to_csv(os.path.join(args.outdir, "robustness_summary.csv"))

    if not args.demo:
        plot_robustness(all_df, args.outdir)
    print(f"\nDone. Outputs in {args.outdir}/")


if __name__ == "__main__":
    main()