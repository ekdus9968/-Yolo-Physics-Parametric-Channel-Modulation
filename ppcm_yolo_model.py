"""
ppcm_yolo_model.py — 길 B: PPCM 을 forward 에 직접 박은 커스텀 YOLOv8
====================================================================
hook 방식 폐기 이유:
  ultralytics .train() 이 model 을 재구성/래핑 → hook 학습중 미적용
  fuse() 가 hook 파괴 → 검증시 소실
  state_dict 에 hook 없음 → 재로드시 소실
  결과: PPCM 이 한 번도 작동 안 함 (baseline 과 동일 값)

해결: BaseModel._predict_once 를 오버라이드해 PPCM 을 forward 에 직접 편입.
  - 첫 입력 x → Stage1
  - Detect 레이어 실행 직전 [N3,N4,N5] → Stage2
  PPCM 이 모델 구조의 일부라 저장·fuse·재로드에 살아남는다.

depth/beta 주입: 커스텀 루프에서 model.ppcm_prepare(img, depth) 를 매 배치 호출.
  forward 는 캐시된 self._ppcm_z, self._ppcm_beta 를 사용.
"""

import torch
import torch.nn as nn
from ultralytics.nn.tasks import DetectionModel

from ppcm_modules import PPCM


class PPCMDetectionModel(DetectionModel):
    """
    DetectionModel + PPCM. _predict_once 를 오버라이드해 Stage1/2 를 삽입.
    stage1_on/stage2_on 토글로 4구성.
    """
    def __init__(self, cfg="yolov8m.yaml", ch=3, nc=None, verbose=True,
                 stage1_on=False, stage2_on=False,
                 learn_gphi=False, n_transmission=2,
                 depth_min=0.5, depth_max=10.0):
        # ★ super().__init__() 이 내부에서 _forward(stride 계산)를 호출하며
        #   우리의 오버라이드된 _predict_once 를 부른다. 그 시점에 아래 속성이
        #   없으면 AttributeError. 그래서 super() 이전에 먼저 설정한다.
        self.stage1_on = stage1_on
        self.stage2_on = stage2_on
        self._ppcm_z = None
        self._ppcm_beta = None
        self.ppcm = None                # 아직 없음 (stride 계산 땐 PPCM 미적용)

        super().__init__(cfg, ch=ch, nc=nc, verbose=verbose)

        # Detect 레이어 인덱스 찾기
        self.detect_idx = len(self.model) - 1
        for i, m in enumerate(self.model):
            if type(m).__name__ == "Detect":
                self.detect_idx = i
                break

        # neck 채널 (Detect 입력 [15,18,21] 채널)
        neck_ch = None
        if stage2_on:
            neck_ch = self._infer_neck_channels()

        # PPCM 을 정식 서브모듈로 (state_dict 에 포함)
        self.ppcm = PPCM(stage1_on=stage1_on, stage2_on=stage2_on,
                         neck_channels=neck_ch,
                         learn_gphi=learn_gphi,
                         n_transmission=n_transmission)
        if stage2_on:
            self.ppcm.stage2.depth_min = depth_min
            self.ppcm.stage2.depth_max = depth_max

    def _infer_neck_channels(self):
        """Detect 입력 3개 레이어의 출력 채널 수를 더미 forward 로 추출."""
        detect = self.model[self.detect_idx]
        f = detect.f  # [15,18,21]
        device = next(self.parameters()).device
        dummy = torch.zeros(1, 3, 256, 256, device=device)
        chans = {}

        # 임시로 순차 실행하며 f 레이어 출력 채널 기록
        y = []
        x = dummy
        with torch.no_grad():
            for m in self.model:
                if m.i >= self.detect_idx:
                    break
                if m.f != -1:
                    x = y[m.f] if isinstance(m.f, int) else \
                        [x if j == -1 else y[j] for j in m.f]
                x = m(x)
                y.append(x if m.i in self.save else None)
                if m.i in f:
                    chans[m.i] = x.shape[1]
        return [chans[i] for i in f]

    def ppcm_prepare(self, img01, depth):
        """
        매 배치 forward 전에 호출. img01: [B,3,H,W] RGB[0,1], depth: [B,1,H,W][0,1].
        s→β 예측 + depth 캐시.
        """
        beta = self.ppcm.prepare(img01, depth)  # PPCM 내부에서 s→β, z 저장
        self._ppcm_beta = beta
        self._ppcm_z = depth
        return beta

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        """
        오버라이드: Stage1(입력) + Stage2(Detect 직전) 삽입.
        depth/beta 는 ppcm_prepare 로 미리 세팅돼 있어야 함(stage on 일 때).
        """
        # --- Stage 1: 입력 이미지에 채널 재가중 ---
        # (self.ppcm 이 아직 None 이면 = 부모 init 의 stride 계산 중 → 건너뜀)
        if self.stage1_on and self.ppcm is not None and self._ppcm_beta is not None:
            x = self.ppcm.stage1(x, self._ppcm_beta)

        y, dt, embeddings = [], [], []
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)

        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else \
                    [x if j == -1 else y[j] for j in m.f]

            # --- Stage 2: Detect 실행 직전, 입력 [N3,N4,N5] 치환 ---
            # (detect_idx 미설정 = 부모 init 중 → getattr 폴백 -1 로 매칭 안 됨)
            if m.i == getattr(self, "detect_idx", -1) and self.stage2_on \
                    and self.ppcm is not None and self._ppcm_z is not None:
                # 이 시점 x = [N3,N4,N5] (Detect.f 로 모인 리스트)
                if isinstance(x, (list, tuple)):
                    x = self.ppcm.stage2(list(x), self._ppcm_z, self._ppcm_beta)

            x = m(x)
            y.append(x if m.i in self.save else None)

            if visualize:
                pass
            if m.i in embed:
                embeddings.append(
                    nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x