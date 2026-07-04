"""
extract_depth.py — Depth Anything V2 로 데이터셋 depth 배치 추출 + 캐싱
======================================================================
기존 dataset.py 의 get_depth() 로직/형식을 그대로 유지:
  - Depth Anything V2 (relative depth, 멀수록 큰 값)
  - min-max 정규화 → [0,1], 0=가까움 1=멀다
  - 저장: (1,H,W) float32 tensor as .pt

우리 데이터셋 구조(RUOD/DUO/Brackish)에 맞춘 배치 버전.
데이터셋별로 캐시 폴더 분리 (파일명 충돌 방지: DUO 의 1.jpg 등).

사용:
  # 전체
  python extract_depth.py --config datasets.yaml --out depth_cache
  # 일부만
  python extract_depth.py --config datasets.yaml --out depth_cache --only RUOD_all DUO
  # 이어하기(이미 있는 캐시 건너뜀) 는 기본 동작

출력 구조:
  depth_cache/
  ├── RUOD_all/
  │   ├── 000009_depth.pt   # (1,H,W) float32 [0,1]
  │   └── ...
  ├── DUO/
  └── ...
"""

import argparse
import glob
import os
import sys
import numpy as np
import torch


def load_config(cfg_path):
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


class DepthExtractor:
    """
    Depth Anything V2 파이프라인 래퍼.
    기존 코드와 동일: pipe(image)['depth'] → np → min-max [0,1] → (1,H,W) tensor.
    """
    def __init__(self, model="depth-anything/Depth-Anything-V2-Small-hf", device=None):
        from transformers import pipeline
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dev_idx = 0 if self.device == "cuda" else -1
        print(f"[depth] loading {model} on {self.device} ...")
        self.pipe = pipeline(task="depth-estimation", model=model, device=dev_idx)

    @torch.no_grad()
    def __call__(self, pil_image):
        res = self.pipe(pil_image)
        d = np.array(res["depth"]).astype(np.float32)   # 멀수록 큼 (depth 방향)
        dmin, dmax = d.min(), d.max()
        if dmax > dmin:
            d = (d - dmin) / (dmax - dmin)              # [0,1], 0=가까움 1=멀다
        else:
            d = np.zeros_like(d)
        return torch.from_numpy(d).unsqueeze(0)         # (1,H,W)


def process_dataset(name, pattern, extractor, out_root, skip_existing=True):
    from PIL import Image
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"  [WARN] no images for {name}: {pattern}", file=sys.stderr)
        return 0, 0
    out_dir = os.path.join(out_root, name)
    os.makedirs(out_dir, exist_ok=True)

    done, skipped = 0, 0
    for i, p in enumerate(paths):
        stem = os.path.splitext(os.path.basename(p))[0]
        cache_pt = os.path.join(out_dir, f"{stem}_depth.pt")
        if skip_existing and os.path.exists(cache_pt):
            skipped += 1
            continue
        try:
            pil = Image.open(p).convert("RGB")
        except Exception as e:
            print(f"    [skip] {p}: {e}", file=sys.stderr)
            continue
        depth = extractor(pil)                 # (1,H,W) [0,1]
        torch.save(depth, cache_pt)
        done += 1
        if (i + 1) % 200 == 0:
            print(f"    [{name}] {i+1}/{len(paths)} (new={done}, skip={skipped})")
    print(f"  [{name}] total={len(paths)}  new={done}  skipped={skipped}")
    return done, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="depth_cache")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--model", default="depth-anything/Depth-Anything-V2-Small-hf")
    ap.add_argument("--device", default=None)
    ap.add_argument("--force", action="store_true", help="기존 캐시 무시하고 재추출")
    args = ap.parse_args()

    cfg = load_config(args.config)
    keys = args.only or list(cfg.keys())
    os.makedirs(args.out, exist_ok=True)

    extractor = DepthExtractor(model=args.model, device=args.device)

    total_new, total_skip = 0, 0
    for name in keys:
        if name not in cfg:
            print(f"  [WARN] {name} not in config", file=sys.stderr)
            continue
        print(f"\n=== {name} ===")
        n, s = process_dataset(name, cfg[name], extractor, args.out,
                               skip_existing=not args.force)
        total_new += n; total_skip += s

    print(f"\n완료. new={total_new}, skipped={total_skip}")
    print(f"캐시 위치: {os.path.abspath(args.out)}/")


if __name__ == "__main__":
    main()