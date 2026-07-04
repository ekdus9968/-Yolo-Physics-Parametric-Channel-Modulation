"""RUOD test split depth 추출."""
import os, glob
import numpy as np
import torch

TEST_GLOB = "data/RUOD/test/images/*.jpg"
OUT_DIR   = "depth_cache/RUOD/test"
MODEL     = "depth-anything/Depth-Anything-V2-Small-hf"

def main():
    from transformers import pipeline
    from PIL import Image

    paths = sorted(glob.glob(TEST_GLOB))
    print(f"RUOD test 이미지: {len(paths)}개")
    if not paths:
        print("[ERROR] 경로 확인")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dev_idx = 0 if device == "cuda" else -1
    print(f"[depth] loading on {device}...")
    pipe = pipeline(task="depth-estimation", model=MODEL, device=dev_idx)

    done = skip = 0
    for i, p in enumerate(paths):
        stem = os.path.splitext(os.path.basename(p))[0]
        cache_pt = os.path.join(OUT_DIR, f"{stem}_depth.pt")
        if os.path.exists(cache_pt):
            skip += 1
            continue
        res = pipe(Image.open(p).convert("RGB"))
        d = np.array(res["depth"]).astype(np.float32)
        dmin, dmax = d.min(), d.max()
        d = (d - dmin) / (dmax - dmin) if dmax > dmin else np.zeros_like(d)
        depth = torch.from_numpy(d).unsqueeze(0)
        torch.save(depth, cache_pt)
        done += 1
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(paths)} (new={done}, skip={skip})")

    print(f"완료. new={done}, skip={skip}")
    print(f"저장: {OUT_DIR}/")

if __name__ == "__main__":
    main()