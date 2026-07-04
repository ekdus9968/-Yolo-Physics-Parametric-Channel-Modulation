"""
extract_duo_test_standalone.py
DUO test split 만 depth 추출. 다른 파일 import 없이 자체 완결.
train 평면 캐시(depth_cache/DUO/*.pt)는 건드리지 않고,
test 는 depth_cache/DUO/test/ 로 저장 (파일명 충돌 방지).
"""
import os, glob
import numpy as np
import torch

# ── 확인: DUO test 이미지 경로 ──
TEST_GLOB = "data/DUO/test/images/*.jpg"
OUT_DIR   = "depth_cache/DUO/test"      # test 전용 하위폴더
MODEL     = "depth-anything/Depth-Anything-V2-Small-hf"
# ────────────────────────────────

def main():
    from transformers import pipeline
    from PIL import Image

    paths = sorted(glob.glob(TEST_GLOB))
    print(f"DUO test 이미지: {len(paths)}개  ({TEST_GLOB})")
    if not paths:
        print("[ERROR] 경로에 이미지 없음. TEST_GLOB 확인.")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dev_idx = 0 if device == "cuda" else -1
    print(f"[depth] loading {MODEL} on {device} ...")
    pipe = pipeline(task="depth-estimation", model=MODEL, device=dev_idx)

    done = skip = 0
    for i, p in enumerate(paths):
        stem = os.path.splitext(os.path.basename(p))[0]
        cache_pt = os.path.join(OUT_DIR, f"{stem}_depth.pt")
        if os.path.exists(cache_pt):
            skip += 1
            continue
        # Depth Anything V2: depth 방향(멀수록 큼), min-max [0,1]
        res = pipe(Image.open(p).convert("RGB"))
        d = np.array(res["depth"]).astype(np.float32)
        dmin, dmax = d.min(), d.max()
        d = (d - dmin) / (dmax - dmin) if dmax > dmin else np.zeros_like(d)
        depth = torch.from_numpy(d).unsqueeze(0)   # (1,H,W)
        torch.save(depth, cache_pt)
        done += 1
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(paths)} (new={done}, skip={skip})")

    print(f"완료. new={done}, skip={skip}")
    print(f"저장 위치: {OUT_DIR}/")

if __name__ == "__main__":
    main()