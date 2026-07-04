#!/usr/bin/env python3
"""
검증 3 — Depth-Color Coupling (Stage 2 물리 전제 확인)
======================================================

목적
----
수중 형성 모델
    I_c(x) = J_c · exp(-β_D^c z) + B_c (1 - exp(-β_D^c z))
이 참이면, "먼 픽셀일수록 파장별로 다르게 감쇠"가 나타나야 한다.
구체적으로 depth z 가 커질수록 log(I_R / I_B) 가 단조 이동(보통 감소)해야 한다.
  → 이것이 Stage2 의 채널별 transmission t_c = exp(-β_D^c z) 의 물리적 근거.

무엇을 뒷받침하나
-----------------
- 실제 데이터(RUOD, Brackish)에서 coupling 존재  → Stage2 정당화
- 합성 데이터(S-UODAC)에서 coupling 부재/역전    → Stage2용 음성 대조군 완성

depth 를 어떻게 얻나 (2단)
--------------------------
- MODE A (기본, 빠름): depth proxy = 어두움/대비 기반 상대 원근.
    수중에서 먼 곳일수록 (a) 어둡고 (b) 대비가 낮고 (c) 배경광에 가까워짐.
    여기서는 간단히 "밝기의 역 + 국소대비 역"을 z 프록시로 사용.
    주의: 이건 근사다. 방향성(상관의 부호/유무) 확인용.
- MODE B (확증, 느림): --depth-dir 로 미리 뽑아둔 WaterMono depth(.npy/.png)를 줌.
    이게 있으면 프록시 대신 실제 z 사용.

핵심 산출
---------
1) 이미지별: depth bin 별 log(R/B) 기울기(회귀계수), Spearman ρ(z, log R/B)
2) 데이터셋별: ρ 분포 → 실제셋은 0에서 유의하게 벗어나고, 합성셋은 ρ≈0
3) fig_depth_color_coupling.png : 데이터셋별 depth-bin vs log(R/B) 곡선
4) fig_coupling_rho_hist.png : 데이터셋별 Spearman ρ 분포

사용
----
python verify3_depth_color.py --config datasets.yaml
python verify3_depth_color.py --config datasets.yaml --depth-dir data/depth_maps  # MODE B
python verify3_depth_color.py --demo
"""

import argparse
import glob
import os
import sys
import numpy as np
import cv2
import pandas as pd
from scipy.stats import spearmanr, ttest_1samp, wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# depth proxy (MODE A)
# ----------------------------------------------------------------------
def depth_proxy(img_bgr):
    """
    상대깊이 프록시 (G 채널 전용).
    검증 대상이 log(R/B) 이므로, proxy 에서 R,B 를 배제하고 G 만 사용해
    '정의상 순환'(같은 채널 공유로 인한 인위적 상관)을 제거한다.
    G 는 수중 중간파장이라 depth 단서(공간 대비 구조)를 담으면서
    R/B 비율과 채널이 겹치지 않는다.
    남는 상관은 오직 'depth 를 공통 원인으로 한 물리적 결합'뿐 —
    이것이 우리가 측정하려는 신호다.
    값이 클수록 '멀다'.
    """
    img = img_bgr.astype(np.float32) / 255.0
    G = img[..., 1]  # 초록만 — R/B 와 채널 독립
    blur = cv2.GaussianBlur(G, (0, 0), sigmaX=4)
    local_contrast = cv2.GaussianBlur((G - blur) ** 2, (0, 0), sigmaX=4)
    local_contrast = np.sqrt(np.maximum(local_contrast, 0))

    def norm(x):
        lo, hi = np.percentile(x, 2), np.percentile(x, 98)
        return np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)

    # 대비 낮을수록 멀다 (산란으로 원거리가 흐려짐)
    z = 1.0 - norm(local_contrast)
    return z

def load_depth_map(depth_path, target_hw):
    """MODE B: 저장된 depth map 로드 후 target 크기로 리사이즈."""
    if depth_path.endswith(".npy"):
        z = np.load(depth_path).astype(np.float32)
    else:
        z = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if z is None:
            return None
        z = z.astype(np.float32)
        if z.ndim == 3:
            z = z.mean(axis=2)
    # 정규화
    z = (z - z.min()) / (z.max() - z.min() + 1e-6)
    z = cv2.resize(z, (target_hw[1], target_hw[0]))
    return z


# ----------------------------------------------------------------------
# 이미지 1장의 coupling 계산
# ----------------------------------------------------------------------
def coupling_for_image(img_bgr, z=None, n_bins=8, eps=1e-4):
    """
    depth bin 별 평균 log(R/B) 를 구하고, z vs log(R/B) 의
    회귀 기울기와 Spearman ρ 를 반환.
    """
    img = img_bgr.astype(np.float32) / 255.0
    B, G, R = img[..., 0], img[..., 1], img[..., 2]
    logRB = np.log((R + eps) / (B + eps))

    if z is None:
        z = depth_proxy(img_bgr)

    zf = z.flatten()
    lf = logRB.flatten()

    # 극단 픽셀(포화/암부) 약간 제거
    valid = (lf > np.percentile(lf, 1)) & (lf < np.percentile(lf, 99))
    zf, lf = zf[valid], lf[valid]
    if len(zf) < 100:
        return None

    # depth bin 별 평균 log(R/B)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(zf, bins) - 1, 0, n_bins - 1)
    bin_means = np.array([lf[idx == b].mean() if (idx == b).any() else np.nan
                          for b in range(n_bins)])

    # 회귀 기울기 (z가 커질 때 logRB 변화율)
    # Spearman(순위 상관)만 사용. 단조성 판정에 충분하고,
    # z 분산이 작아도 수치적으로 안정적.
    # (과거의 polyfit slope 는 z가 거의 상수인 이미지에서 poorly-conditioned
    #  경고를 냈고 최종 판정에도 쓰이지 않아 제거함.)
    rho, _ = spearmanr(zf, lf)

    return {
        "spearman_rho": float(rho),
        "bin_means": bin_means,
    }


def process_dataset(name, paths, depth_dir=None, max_images=300):
    if max_images:
        paths = paths[:max_images]
    rows, bin_curves = [], []
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
                    z = load_depth_map(dp, img.shape[:2])
                    break
        res = coupling_for_image(img, z=z)
        if res is None:
            continue
        rows.append({"dataset": name, "path": p,
                     "spearman_rho": res["spearman_rho"]})
        bin_curves.append(res["bin_means"])
    df = pd.DataFrame(rows)
    curves = np.array(bin_curves) if bin_curves else np.zeros((0, 8))
    print(f"  [{name}] {len(df)} images, mean ρ = {df['spearman_rho'].mean():.3f}"
          if len(df) else f"  [{name}] no valid images")
    return df, curves


# ----------------------------------------------------------------------
# 시각화
# ----------------------------------------------------------------------
def plot_coupling_curves(curves_by_ds, out_path):
    """데이터셋별 depth-bin vs log(R/B) 평균 곡선."""
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("tab10")
    x = np.linspace(0, 1, 8)
    for i, (name, curves) in enumerate(sorted(curves_by_ds.items())):
        if len(curves) == 0:
            continue
        mean_curve = np.nanmean(curves, axis=0)
        std_curve = np.nanstd(curves, axis=0)
        c = cmap(i % 10)
        ax.plot(x, mean_curve, "-o", color=c, label=name, lw=2)
        ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve,
                        color=c, alpha=0.15)
    ax.set_xlabel("relative depth z  (0=near, 1=far)")
    ax.set_ylabel("mean log(R/B)")
    ax.set_title("Depth–color coupling per dataset\n"
                 "(monotone decreasing = physical: far pixels lose red faster)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_rho_hist(all_df, out_path):
    """데이터셋별 Spearman ρ 분포."""
    datasets = sorted(all_df["dataset"].unique())
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, d in enumerate(datasets):
        vals = all_df[all_df["dataset"] == d]["spearman_rho"].dropna()
        ax.hist(vals, bins=30, alpha=0.5, color=cmap(i % 10),
                label=f"{d} (μ={vals.mean():+.2f})", density=True)
    ax.axvline(0, color="black", ls="--", lw=1, label="ρ=0 (no coupling)")
    ax.set_xlabel("Spearman ρ ( z  vs  log(R/B) )  per image")
    ax.set_ylabel("density")
    ax.set_title("Depth–color coupling strength\n"
                 "(mass away from 0 = real physical coupling; centered at 0 = none)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  saved {out_path}")


# ----------------------------------------------------------------------
# demo
# ----------------------------------------------------------------------
def make_demo_image(coupled=True, seed=0):
    """coupled=True: 물리적 depth-color 결합 있음. False: 랜덤(합성 흉내)."""
    rng = np.random.default_rng(seed)
    H, W = 120, 160
    J = rng.uniform(0.4, 0.9, size=(H, W, 3)).astype(np.float32)
    z = np.linspace(0.1, 1.0, H)[:, None] * np.ones((1, W))
    z = (z + rng.normal(0, 0.05, (H, W))).clip(0, 1).astype(np.float32)
    out = np.zeros((H, W, 3), np.float32)
    if coupled:
        betas = [0.2, 0.5, 1.2]  # BGR: R(맨끝) 가장 크게 감쇠
        Binf = [0.15, 0.12, 0.08]
        for c in range(3):
            t = np.exp(-betas[c] * z * 3)
            out[..., c] = J[..., c] * t + Binf[c] * (1 - t)
    else:
        gains = rng.uniform(0.5, 1.5, 3)  # depth 무관 랜덤
        for c in range(3):
            out[..., c] = J[..., c] * gains[c]
    return (out.clip(0, 1) * 255).astype(np.uint8)


def run_demo():
    print("[DEMO] coupled vs uncoupled synthetic sets...")
    curves_by_ds = {}
    frames = []
    for name, coupled in [("demo_real_coupled", True), ("demo_synthetic_random", False)]:
        rows, curves = [], []
        for i in range(80):
            img = make_demo_image(coupled=coupled, seed=(i if coupled else 500 + i))
            res = coupling_for_image(img)
            if res:
                rows.append({"dataset": name, "path": f"{name}_{i}",
                             "spearman_rho": res["spearman_rho"]})
                curves.append(res["bin_means"])
        frames.append(pd.DataFrame(rows))
        curves_by_ds[name] = np.array(curves)
        print(f"  [{name}] mean ρ = {frames[-1]['spearman_rho'].mean():+.3f}")
    return pd.concat(frames, ignore_index=True), curves_by_ds


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
    ap.add_argument("--depth-dir", help="MODE B: 미리 뽑은 depth map 폴더")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--max-images", type=int, default=300)
    ap.add_argument("--outdir", default="verify3_out")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.demo or not args.config:
        all_df, curves_by_ds = run_demo()
    else:
        cfg = load_config(args.config)
        frames, curves_by_ds = [], {}
        for name, pattern in cfg.items():
            paths = sorted(glob.glob(pattern))
            if not paths:
                print(f"  [WARN] no images for {name}: {pattern}", file=sys.stderr)
                continue
            df, curves = process_dataset(name, paths, args.depth_dir, args.max_images)
            frames.append(df)
            curves_by_ds[name] = curves
        if not frames:
            print("No data.", file=sys.stderr); sys.exit(1)
        all_df = pd.concat(frames, ignore_index=True)

    all_df.to_csv(os.path.join(args.outdir, "coupling_per_image.csv"), index=False)

    # 요약
    print("\n=== Per-dataset depth-color coupling ===")
    summary = all_df.groupby("dataset")["spearman_rho"].agg(["mean", "std", "count"])
    print(summary.round(3).to_string())
    print("\n해석: |mean ρ| 이 크고 0에서 멀면 → 물리적 coupling 존재 (실제 수중).")
    print("      mean ρ ≈ 0 이면 → coupling 없음 (합성/무관).")
    summary.to_csv(os.path.join(args.outdir, "coupling_summary.csv"))

    # ── 이미지 단위 유의성 검정 ────────────────────────────────────
    # 분석 단위 = 이미지 (픽셀 아님). 각 이미지의 ρ 를 표본으로,
    # 이 표본 평균이 0과 유의하게 다른지 검정.
    #   - t-test(모수) + Wilcoxon(비모수) 병행
    #   - 효과크기 d = mean/std 함께 보고 (유의하지만 작은 경우 구분)
    print("\n=== Image-level significance (ρ != 0?) ===")
    stat_rows = []
    for d in sorted(all_df["dataset"].unique()):
        rhos = all_df[all_df.dataset == d]["spearman_rho"].dropna().values
        if len(rhos) < 5:
            print(f"  {d:24} n={len(rhos)} (too few)")
            continue
        t, pt = ttest_1samp(rhos, 0.0)
        try:
            w, pw = wilcoxon(rhos)
        except ValueError:
            w, pw = np.nan, np.nan
        eff = rhos.mean() / (rhos.std(ddof=1) + 1e-9)  # Cohen's d vs 0
        print(f"  {d:24} n={len(rhos):4d}  mean_rho={rhos.mean():+.3f}  "
              f"d={eff:+.2f}  t-p={pt:.1e}  wilcoxon-p={pw:.1e}")
        stat_rows.append({"dataset": d, "n": len(rhos),
                          "mean_rho": rhos.mean(), "cohens_d": eff,
                          "ttest_p": pt, "wilcoxon_p": pw})
    pd.DataFrame(stat_rows).to_csv(
        os.path.join(args.outdir, "coupling_significance.csv"), index=False)
    print("  → 실제셋: |d| 크고 p 작음(유의). 합성셋: mean_rho≈0, d≈0.")
    print("  (주의: 픽셀 단위가 아닌 이미지 단위 검정이라 통계적으로 타당)")

    plot_coupling_curves(curves_by_ds, os.path.join(args.outdir, "fig_depth_color_coupling.png"))
    plot_rho_hist(all_df, os.path.join(args.outdir, "fig_coupling_rho_hist.png"))
    print(f"\nDone. Outputs in {args.outdir}/")


if __name__ == "__main__":
    main()