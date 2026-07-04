"""DUO test split 만 depth 추출 (train 평면 캐시는 그대로 둠)."""
import os, glob, torch
from PIL import Image
from extract_depth import DepthExtractor, cache_path_for

# ── 여기만 확인: DUO test 이미지 경로 ──
TEST_GLOB = "data/DUO/test/images/*.jpg"
OUT_ROOT  = "depth_cache"
DATASET   = "DUO"   # 캐시가 depth_cache/DUO/test/ 로 가도록 key 는 DUO 유지
# ────────────────────────────────────

paths = sorted(glob.glob(TEST_GLOB))
print(f"DUO test 이미지: {len(paths)}개")
if not paths:
    print("경로 확인 필요:", TEST_GLOB)
    raise SystemExit

ext = DepthExtractor()  # Depth Anything V2
done = skip = 0
for i, p in enumerate(paths):
    out_dir, cache_pt = cache_path_for(OUT_ROOT, DATASET, p)  # → depth_cache/DUO/test/
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(cache_pt):
        skip += 1; continue
    depth = ext(Image.open(p).convert("RGB"))
    torch.save(depth, cache_pt)
    done += 1
    if (i+1) % 200 == 0:
        print(f"  {i+1}/{len(paths)} (new={done})")
print(f"완료. new={done}, skip={skip}")
print(f"저장 위치: {OUT_ROOT}/{DATASET}/test/")