"""
train_ppcm.py — PPCM + YOLOv8 학습 배선 (4개 토글 구성)
=======================================================
forward-hook 방식으로 순정 YOLOv8 에 PPCM 부착.
depth 는 extract_depth.py 로 미리 캐싱한 .pt 를 로드.

4개 구성:
  baseline : --stage1 0 --stage2 0
  stage1   : --stage1 1 --stage2 0
  stage2   : --stage1 0 --stage2 1
  full     : --stage1 1 --stage2 1

주의: 이 스크립트는 '배선 골격'이다. ultralytics 학습 루프에 PPCM 을
      끼우는 방식은 두 가지가 있어 아래 두 경로를 모두 제공한다:
  (A) 커스텀 학습 루프 (권장, 완전 제어): depth 를 배치마다 주입
  (B) ultralytics Trainer 콜백 (간편하지만 depth 주입에 커스텀 필요)
여기서는 (A) 의 핵심 배선을 보인다. 실제 데이터로더/augmentation 은
사용자 환경의 기존 dataset.py 와 연결.
"""

import argparse
import os
import glob
import numpy as np
import torch
import torch.nn as nn
from ppcm_modules import PPCM, attach_ppcm_hooks


# =====================================================================
# depth 캐시 로더 (기존 형식 유지: .pt, (1,H,W), [0,1])
# =====================================================================
class DepthCache:
    """
    extract_depth.py 가 만든 depth 캐시를 로드.
    split 하위폴더 구조 지원: depth_cache/<dataset>/<split>/<stem>_depth.pt
    (DUO 처럼 train/test 에 동일 파일명이 있어도 경로로 구분)
    """
    def __init__(self, cache_root, dataset_name):
        self.root = os.path.join(cache_root, dataset_name)
        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"depth cache 없음: {self.root} "
                                    f"(먼저 extract_depth.py 실행)")

    @staticmethod
    def _infer_split(img_path):
        parts = img_path.replace("\\", "/").lower().split("/")
        for key in ("train", "test", "val", "valid"):
            if key in parts:
                return key
        for p in parts:
            if p.startswith("train"):
                return "train"
            if p.startswith("test") or p.startswith("val"):
                return "test"
        return ""

    def _resolve(self, img_path):
        """이미지 경로 → 캐시 .pt 경로. split 하위폴더 우선, 없으면 평면 구조 폴백."""
        stem = os.path.splitext(os.path.basename(img_path))[0]
        split = self._infer_split(img_path)
        candidates = []
        if split:
            candidates.append(os.path.join(self.root, split, f"{stem}_depth.pt"))
        candidates.append(os.path.join(self.root, f"{stem}_depth.pt"))  # 평면 폴백
        for c in candidates:
            if os.path.exists(c):
                return c
        return candidates[0]  # 없으면 첫 후보(에러 메시지용)

    def get(self, img_path, target_hw=None):
        pt = self._resolve(img_path)
        if not os.path.exists(pt):
            raise FileNotFoundError(f"depth 캐시 미스: {pt}")
        d = torch.load(pt)                      # (1,H,W) [0,1]
        if d.dim() == 2:
            d = d.unsqueeze(0)
        if target_hw is not None and d.shape[-2:] != tuple(target_hw):
            d = torch.nn.functional.interpolate(
                d.unsqueeze(0), size=target_hw, mode="bilinear",
                align_corners=False).squeeze(0)
        return d                                # (1,H,W)

    def get_batch(self, img_paths, target_hw, device):
        """배치 이미지 경로 리스트 → [B,1,H,W]."""
        ds = [self.get(p, target_hw) for p in img_paths]
        return torch.stack(ds, dim=0).to(device)


# =====================================================================
# neck 채널 수 자동 추출 (YOLOv8m 등)
# =====================================================================
def get_neck_channels(yolo_model, detect_index=None):
    """
    Detect 입력 3개 스케일(N3,N4,N5)의 채널 수를 추출.
    forward-pre-hook 으로 실제 텐서 shape 을 한 번 잡아서 확인.
    """
    seq = yolo_model.model
    if detect_index is None:
        detect_index = len(seq) - 1
        for i, m in enumerate(seq):
            if type(m).__name__ == "Detect":
                detect_index = i; break

    captured = {}
    def cap_hook(module, args):
        feats = args[0]
        if isinstance(feats, (list, tuple)):
            captured["ch"] = [f.shape[1] for f in feats]
        return None
    h = seq[detect_index].register_forward_pre_hook(cap_hook)
    # 더미 forward 로 shape 확보
    device = next(yolo_model.parameters()).device
    dummy = torch.zeros(1, 3, 640, 640, device=device)
    was_training = yolo_model.training
    yolo_model.eval()
    with torch.no_grad():
        try:
            yolo_model(dummy)
        except Exception:
            pass
    if was_training:
        yolo_model.train()
    h.remove()
    return captured.get("ch", None), detect_index


# =====================================================================
# PPCM 부착 헬퍼
# =====================================================================
def build_ppcm_for_model(yolo_model, stage1_on, stage2_on,
                         learn_gphi=False, n_transmission=2,
                         depth_min=0.5, depth_max=10.0):
    """
    모델에서 neck 채널 자동 추출 후 PPCM 생성 + hook 부착.
    반환: (ppcm, hook_handles, detect_index)
    """
    neck_ch, detect_index = get_neck_channels(yolo_model)
    if stage2_on and neck_ch is None:
        raise RuntimeError("neck 채널 추출 실패 — detect_index 수동 지정 필요")
    if stage2_on:
        print(f"[ppcm] neck channels (N3,N4,N5) = {neck_ch}")

    ppcm = PPCM(stage1_on=stage1_on, stage2_on=stage2_on,
                neck_channels=neck_ch if stage2_on else None,
                learn_gphi=learn_gphi, n_transmission=n_transmission)
    # depth 미터 스케일 반영
    if stage2_on:
        ppcm.stage2.depth_min = depth_min
        ppcm.stage2.depth_max = depth_max

    handles = attach_ppcm_hooks(yolo_model, ppcm, detect_index=detect_index)
    return ppcm, handles, detect_index


# =====================================================================
# 학습 루프 골격 (경로 A: 커스텀)
# =====================================================================
def train_skeleton(args):
    """
    핵심 배선만. 실제 데이터로더/loss 는 ultralytics 것을 쓰거나
    기존 dataset.py 와 연결.
    """
    from ultralytics import YOLO

    # 1) 순정 YOLOv8 로드
    yolo = YOLO(args.weights)          # 예: 'yolov8m.pt'
    model = yolo.model                 # DetectionModel (nn.Module)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # 2) PPCM 부착 (토글)
    ppcm, handles, det_idx = build_ppcm_for_model(
        model, bool(args.stage1), bool(args.stage2),
        learn_gphi=bool(args.learn_gphi),
        n_transmission=args.n_transmission,
        depth_min=args.depth_min, depth_max=args.depth_max)
    ppcm.to(device)

    # 3) depth 캐시
    depth_cache = DepthCache(args.depth_cache, args.dataset_name) \
        if (args.stage1 or args.stage2) else None

    # 4) optimizer: YOLO 파라미터 + PPCM 파라미터
    ppcm_params = [p for p in ppcm.parameters() if p.requires_grad]
    print(f"[train] PPCM learnable params: {sum(p.numel() for p in ppcm_params)}")
    # (실제로는 ultralytics optimizer 에 ppcm_params 를 add_param_group 로 추가)

    print(f"""
[준비 완료] 구성 요약
  config     : stage1={args.stage1}, stage2={args.stage2}
  weights    : {args.weights}
  detect_idx : {det_idx}
  depth      : {'ON (cache=' + args.depth_cache + '/' + args.dataset_name + ')' if depth_cache else 'OFF'}
  device     : {device}

다음: 실제 학습 루프에서 매 배치마다
    ppcm.prepare(img_rgb01, depth_batch)   # depth_batch = depth_cache.get_batch(paths, hw, device)
  를 호출한 뒤 model(img) 실행하면 hook 이 자동으로 PPCM 적용.
  baseline 이면 prepare 호출 없이도 hook 이 identity 로 통과.
""")

    # hook 제거 (정리)
    # for h in handles: h.remove()
    return ppcm, model, depth_cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolov8m.pt")
    ap.add_argument("--stage1", type=int, default=0, choices=[0, 1])
    ap.add_argument("--stage2", type=int, default=0, choices=[0, 1])
    ap.add_argument("--learn-gphi", type=int, default=0, choices=[0, 1],
                    help="g_φ 학습(Mode2). 0=휴리스틱(Mode1)")
    ap.add_argument("--n-transmission", type=int, default=2, choices=[2, 3])
    ap.add_argument("--depth-cache", default="depth_cache")
    ap.add_argument("--dataset-name", default="RUOD_all")
    ap.add_argument("--depth-min", type=float, default=0.5)
    ap.add_argument("--depth-max", type=float, default=10.0)
    args = ap.parse_args()

    cfg_name = {(0,0):"baseline", (1,0):"stage1",
                (0,1):"stage2", (1,1):"full"}[(args.stage1, args.stage2)]
    print(f"=== PPCM 구성: {cfg_name} ===")
    train_skeleton(args)


if __name__ == "__main__":
    main()