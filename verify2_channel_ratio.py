#!/usr/bin/env python3
"""
검증 2 — 데이터셋별 물리 신호(채널비율) 실재 확인
================================================================

목적
----
PPCM의 전제: 관측된 색 편이가 파장별 감쇠 β_D^c 와 거리 z 의 물리적 곱
    I_c = J_c · exp(-β_D^c z) + B_c (1 - exp(-β_D^c z))
에서 나왔다는 것. 이 전제가 각 데이터셋에서 참인지 확인한다.

왜 "채널비율"인가 (depth-robustness)
------------------------------------
red-channel 절대값은 β_R 과 z 가 섞여(교락) 있다.
비율 log(I_R / I_B) 를 쓰면 공통 z 항이 상당 부분 상쇄되어
"water type의 스펙트럼 특성"을 depth보다 직접 반영한다.
    근사적으로 log(I_R/I_B) ~ (β_B - β_R)·z + (색/조명 항)
    → 채널 간 감쇠 차이(스펙트럼 기울기)가 신호로 남음.

왜 "근거리 영역만"인가
----------------------
depth 영향을 더 줄이기 위해, 각 이미지에서 상위 밝기 픽셀(근거리 추정)만 사용.
멀리 있는 배경일수록 backscatter가 지배해 water-type 지문이 흐려지므로,
근거리 표본이 β_D 스펙트럼을 더 깨끗하게 담는다.

기대 결과 (3단 데이터셋 구성의 정당화)
--------------------------------------
- RUOD 내부 clear vs turbid subset: 채널비율 분포가 유의하게 갈림   → Stage1 정당화
- DUO(맑은 근해) vs Brackish(탁한 기수역): 극단적으로 갈림          → cross-domain 대비
- S-UODAC2020의 합성 type들: 물리적으로 안 갈림(뭉치거나 비물리적) → 음성 대조군

출력
----
1) datasets/<name>_ratios.csv : 이미지별 채널비율 통계
2) fig_channel_ratio_distributions.png : 데이터셋별 분포 겹쳐그리기 (논문 figure 후보)
3) fig_channel_ratio_scatter.png : (log R/G, log R/B) 산점도 - 수형 클러스터 가시화
4) summary.csv : 데이터셋별 요약 통계 + 분포 분리도(effect size)

사용법
------
python verify2_channel_ratio.py --config datasets.yaml
  datasets.yaml 예시:
    RUOD_clear:  /path/to/ruod/clear/*.jpg
    RUOD_turbid: /path/to/ruod/turbid/*.jpg
    DUO:         /path/to/duo/images/*.jpg
    Brackish:    /path/to/brackish/*.jpg
    S-UODAC_type1: /path/to/suodac/type1/*.jpg
    ...
또는 --demo 로 합성 데이터에서 로직 검증만 수행.
"""

import argparse
import glob
import os
import sys
import numpy as np
import cv2
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# 핵심 계산
# ----------------------------------------------------------------------
def compute_channel_ratios(img_bgr, bright_percentile=80, eps=1e-4):
    """
    한 이미지에서 근거리(상위 밝기) 영역의 채널비율 통계를 계산.

    Parameters
    ----------
    img_bgr : (H,W,3) uint8, OpenCV BGR
    bright_percentile : 근거리로 간주할 밝기 상위 백분위수 (기본 상위 20%)
    eps : log 안정화

    Returns
    -------
    dict: log_RB, log_RG, log_GB (근거리 영역 평균), 및 밝기
    """
    img = img_bgr.astype(np.float32) / 255.0
    B, G, R = img[..., 0], img[..., 1], img[..., 2]

    # 밝기(근거리 프록시): 채널 평균 밝기
    lum = (R + G + B) / 3.0

    # 상위 밝기 픽셀 마스크 = 근거리 추정
    thr = np.percentile(lum, bright_percentile)
    mask = lum >= thr
    if mask.sum() < 50:  # 너무 적으면 전체 사용
        mask = np.ones_like(lum, dtype=bool)

    Rm = R[mask].mean()
    Gm = G[mask].mean()
    Bm = B[mask].mean()

    return {
        "log_RB": float(np.log((Rm + eps) / (Bm + eps))),  # R vs B 감쇠 기울기
        "log_RG": float(np.log((Rm + eps) / (Gm + eps))),
        "log_GB": float(np.log((Gm + eps) / (Bm + eps))),
        "mean_lum": float(lum.mean()),
        "R_near": float(Rm),
        "G_near": float(Gm),
        "B_near": float(Bm),
    }


def process_dataset(name, paths, max_images=None):
    """한 데이터셋의 모든 이미지를 처리해 DataFrame 반환."""
    if max_images:
        paths = paths[:max_images]
    rows = []
    n_fail = 0
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            n_fail += 1
            continue
        stats = compute_channel_ratios(img)
        stats["path"] = p
        stats["dataset"] = name
        rows.append(stats)
    if n_fail:
        print(f"  [{name}] {n_fail} images failed to load", file=sys.stderr)
    df = pd.DataFrame(rows)
    print(f"  [{name}] processed {len(df)} images")
    return df


# ----------------------------------------------------------------------
# 분리도(effect size) — 두 분포가 얼마나 갈리는지
# ----------------------------------------------------------------------
def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled == 0:
        return np.nan
    return (a.mean() - b.mean()) / pooled


def build_d_matrix(df, metric="log_RB"):
    datasets = sorted(df["dataset"].unique())
    n = len(datasets)
    mat = np.zeros((n, n))
    for i, di in enumerate(datasets):
        for j, dj in enumerate(datasets):
            if i == j:
                mat[i, j] = 0.0
                continue
            a = df[df["dataset"] == di][metric].dropna()
            b = df[df["dataset"] == dj][metric].dropna()
            mat[i, j] = cohens_d(a, b)
    return datasets, mat


def plot_heatmap(datasets, mat, metric, out_path, clip=None):
    """
    clip: 컬러스케일 절대값 상한 (예: 3). None이면 자동.
    |d|가 워낙 크게 벌어져 있어서(최대 46) clip 안 하면
    대부분 칸이 다 같은 색으로 뭉개져 보일 수 있음.
    """
    n = len(datasets)
    plot_mat = mat.copy()
    if clip is not None:
        plot_mat = np.clip(plot_mat, -clip, clip)

    fig, ax = plt.subplots(figsize=(1.1 * n + 2, 1.1 * n + 1))
    im = ax.imshow(plot_mat, cmap="RdBu_r", vmin=-(clip or np.nanmax(np.abs(mat))),
                   vmax=(clip or np.nanmax(np.abs(mat))))

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(datasets, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(datasets, fontsize=9)

    # 각 칸에 실제(클리핑 전) 값 표기
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            val = mat[i, j]
            color = "white" if abs(plot_mat[i, j]) > (clip or np.nanmax(np.abs(mat))) * 0.6 else "black"
            ax.text(j, i, f"{val:+.1f}", ha="center", va="center",
                    fontsize=7, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(f"Cohen's d ({'clipped at ±%d' % clip if clip else 'raw'})")

    title_suffix = f" (clipped ±{clip} for readability)" if clip else ""
    ax.set_title(f"Pairwise separation on {metric}  [Cohen's d]{title_suffix}\n"
                 "|d|>0.8 = LARGE separation (distinct water type)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"saved {out_path}")




# ----------------------------------------------------------------------
# 시각화
# ----------------------------------------------------------------------
def plot_distributions(all_df, out_path):
    """데이터셋별 log_RB 분포 겹쳐그리기 (KDE-ish 히스토그램)."""
    metrics = ["log_RB", "log_RG", "log_GB"]
    titles = {
        "log_RB": "log(R/B)  —  red-blue attenuation slope",
        "log_RG": "log(R/G)",
        "log_GB": "log(G/B)",
    }
    datasets = sorted(all_df["dataset"].unique())
    cmap = plt.get_cmap("tab10")
    colors = {d: cmap(i % 10) for i, d in enumerate(datasets)}

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, m in zip(axes, metrics):
        for d in datasets:
            vals = all_df[all_df["dataset"] == d][m].dropna().values
            if len(vals) == 0:
                continue
            ax.hist(vals, bins=40, density=True, alpha=0.45,
                    color=colors[d], label=d)
        ax.set_title(titles[m], fontsize=11)
        ax.set_xlabel(m)
        ax.set_ylabel("density")
        ax.axvline(0, color="gray", ls=":", lw=0.8)
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Verification 2 — Channel-ratio physics signal per dataset\n"
                 "(near-field pixels only; separated distributions = real optical water-type difference)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_scatter(all_df, out_path):
    """(log R/G, log R/B) 2D 산점도 — 수형 클러스터 가시화."""
    datasets = sorted(all_df["dataset"].unique())
    cmap = plt.get_cmap("tab10")
    colors = {d: cmap(i % 10) for i, d in enumerate(datasets)}

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for d in datasets:
        sub = all_df[all_df["dataset"] == d]
        ax.scatter(sub["log_RG"], sub["log_RB"], s=10, alpha=0.5,
                   color=colors[d], label=d)
        # 클러스터 중심
        ax.scatter(sub["log_RG"].mean(), sub["log_RB"].mean(),
                   s=220, marker="X", color=colors[d],
                   edgecolor="black", linewidth=1.5, zorder=5)
    ax.set_xlabel("log(R/G)")
    ax.set_ylabel("log(R/B)")
    ax.set_title("Water-type fingerprint in channel-ratio space\n"
                 "(X = dataset centroid; separated centroids = distinct β_D spectra)")
    ax.axhline(0, color="gray", ls=":", lw=0.8)
    ax.axvline(0, color="gray", ls=":", lw=0.8)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  saved {out_path}")


# ----------------------------------------------------------------------
# 합성 데모 데이터 (환경에 실제 데이터 없을 때 로직 검증용)
# ----------------------------------------------------------------------
def make_demo_image(beta_r, beta_g, beta_b, seed=0):
    """
    간이 수중 형성 모델로 합성 이미지 생성.
    beta 클수록 그 채널이 거리에 따라 더 죽음 → water type 흉내.
    """
    rng = np.random.default_rng(seed)
    H, W = 120, 160
    # 장면 반사율 J (랜덤 텍스처)
    J = rng.uniform(0.3, 0.9, size=(H, W, 3)).astype(np.float32)
    # depth: 위→아래로 멀어지는 그라디언트 + 노이즈
    z = np.linspace(0.3, 3.0, H)[:, None] * np.ones((1, W))
    z = (z + rng.normal(0, 0.15, size=(H, W))).clip(0.1, 4.0).astype(np.float32)
    betas = np.array([beta_b, beta_g, beta_r], dtype=np.float32)  # BGR 순
    B_inf = np.array([0.15, 0.12, 0.08], dtype=np.float32)        # 배경광 BGR
    out = np.zeros((H, W, 3), np.float32)
    for c in range(3):
        t = np.exp(-betas[c] * z)
        out[..., c] = J[..., c] * t + B_inf[c] * (1 - t)
    return (out.clip(0, 1) * 255).astype(np.uint8)


def run_demo():
    """
    합성 데이터로 3개 가짜 '데이터셋'을 만들어 스크립트 로직을 검증.
    - fake_clear:  β_R 크고 β_B 작음 (맑은 파란 물: 빨강만 죽음) → log(R/B) 매우 음수
    - fake_turbid: β 전체 크고 균등 (탁한 물)                    → log(R/B) 덜 음수
    - fake_synthetic: 물리 무관 랜덤 색 이동 (WCT2 흉내)         → 비물리적 분산
    """
    print("[DEMO] generating synthetic datasets to validate logic...")
    frames = []

    # fake_clear: 파란 맑은 물
    rows = []
    for i in range(60):
        img = make_demo_image(beta_r=0.9, beta_g=0.3, beta_b=0.05, seed=i)
        s = compute_channel_ratios(img); s["dataset"] = "fake_clear"; s["path"] = f"clear_{i}"
        rows.append(s)
    frames.append(pd.DataFrame(rows))

    # fake_turbid: 탁한 물 (전 채널 감쇠 큼, 녹색조)
    rows = []
    for i in range(60):
        img = make_demo_image(beta_r=0.7, beta_g=0.5, beta_b=0.45, seed=100 + i)
        s = compute_channel_ratios(img); s["dataset"] = "fake_turbid"; s["path"] = f"turbid_{i}"
        rows.append(s)
    frames.append(pd.DataFrame(rows))

    # fake_synthetic: 물리 무관 (랜덤 채널 게인 = WCT2류 합성)
    rows = []
    rng = np.random.default_rng(7)
    for i in range(60):
        base = make_demo_image(beta_r=0.5, beta_g=0.5, beta_b=0.5, seed=200 + i)
        gains = rng.uniform(0.5, 1.5, size=3)  # 물리와 무관한 임의 채널 게인
        synth = (base.astype(np.float32) * gains[None, None, :]).clip(0, 255).astype(np.uint8)
        s = compute_channel_ratios(synth); s["dataset"] = "fake_synthetic"; s["path"] = f"synth_{i}"
        rows.append(s)
    frames.append(pd.DataFrame(rows))

    all_df = pd.concat(frames, ignore_index=True)
    return all_df


# ----------------------------------------------------------------------
# 요약 통계
# ----------------------------------------------------------------------
def build_summary(all_df):
    datasets = sorted(all_df["dataset"].unique())
    summary = all_df.groupby("dataset")[["log_RB", "log_RG", "log_GB", "mean_lum"]].agg(
        ["mean", "std"]
    )
    print("\n=== Per-dataset summary (channel ratios, near-field) ===")
    print(summary.round(3).to_string())

    # 쌍별 분리도 (log_RB 기준)
    print("\n=== Pairwise separation on log(R/B)  [Cohen's d] ===")
    print("  |d|>0.8 = large (물리적으로 뚜렷이 갈림), <0.2 = negligible (안 갈림)")
    sep_rows = []
    for i in range(len(datasets)):
        for j in range(i + 1, len(datasets)):
            a = all_df[all_df["dataset"] == datasets[i]]["log_RB"].dropna()
            b = all_df[all_df["dataset"] == datasets[j]]["log_RB"].dropna()
            d = cohens_d(a, b)
            tag = "LARGE" if abs(d) > 0.8 else ("small" if abs(d) > 0.2 else "~none")
            print(f"  {datasets[i]:>18} vs {datasets[j]:<18}: d = {d:+.2f}  [{tag}]")
            sep_rows.append({"pair": f"{datasets[i]} vs {datasets[j]}",
                             "cohens_d_logRB": d, "magnitude": tag})
    return summary, pd.DataFrame(sep_rows)


def load_config(cfg_path):
    """아주 단순한 key: glob-pattern 형식 파서 (yaml 의존 없이)."""
    mapping = {}
    with open(cfg_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, pat = line.split(":", 1)
            mapping[key.strip()] = pat.strip()
    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="datasets config (key: glob-pattern per line)")
    ap.add_argument("--demo", action="store_true", help="run on synthetic data")
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--outdir", default="verify2_out")
    
    ap.add_argument("--csv", default="verify2_out/all_ratios.csv")
    ap.add_argument("--metric", default="log_RB", choices=["log_RB", "log_RG", "log_GB"])
    ap.add_argument("--clip", type=float, default=3.0,
                     help="colorbar clip range (raw values still printed in cells)")
    ap.add_argument("--out", default="verify2_out/fig_separation_heatmap.png")

    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.csv)
    datasets, mat = build_d_matrix(df, metric=args.metric)
    plot_heatmap(datasets, mat, args.metric, args.out, clip=args.clip)

    if args.demo or not args.config:
        all_df = run_demo()
    else:
        cfg = load_config(args.config)
        frames = []
        for name, pattern in cfg.items():
            paths = sorted(glob.glob(pattern))
            if not paths:
                print(f"  [WARN] no images for {name}: {pattern}", file=sys.stderr)
                continue
            frames.append(process_dataset(name, paths, args.max_images))
        if not frames:
            print("No data found. Check config paths.", file=sys.stderr)
            sys.exit(1)
        all_df = pd.concat(frames, ignore_index=True)

    # 저장
    all_df.to_csv(os.path.join(args.outdir, "all_ratios.csv"), index=False)
    summary, sep = build_summary(all_df)
    summary.to_csv(os.path.join(args.outdir, "summary.csv"))
    sep.to_csv(os.path.join(args.outdir, "separation.csv"), index=False)

    # 그림
    plot_distributions(all_df, os.path.join(args.outdir, "fig_channel_ratio_distributions.png"))
    plot_scatter(all_df, os.path.join(args.outdir, "fig_channel_ratio_scatter.png"))

    print(f"\nDone. Outputs in {args.outdir}/")


if __name__ == "__main__":
    main()