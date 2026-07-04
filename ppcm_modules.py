"""
ppcm_modules.py — PPCM (Physics-Prior Confidence Modulation)
=============================================================
PPCM_SPEC.md 와 1:1 대응하는 구현.  forward-hook 방식, 4개 토글.

SPEC 대응 표시: 각 클래스/메서드에 [P1]..[P7], [N1]..[N5] 로 근거를 단다.

구성:
  - PhysInterp        : s(turbidity 좌표) → β_D^c  (6점 앵커 보간)  [P5]
  - WaterTypePredictor: g_φ. Mode1(휴리스틱/고정) / Mode2(학습)      [3장]
  - PPCMStage1        : 채널 residual, 평균중심                      [P1][N3]
  - PPCMStage2        : 채널별 transmission, zero-init γ, centering  [P2][P3][P4][N4]
  - PPCM              : 토글 컨테이너 + forward-hook 부착 헬퍼         [P6][P7]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# PhysInterp — s → β_D^c  (β_D 는 고정 prior, gradient 안 받음)  [P5]
# =====================================================================
class PhysInterp(nn.Module):
    """
    Solonenko & Mobley 6개 water-type 의 (β_R,β_G,β_B) 를 앵커로 두고,
    저차원 좌표 s∈[0,1] 로 그 사이를 monotone 보간해 β_D^c 를 산출.
    - 앵커 값은 buffer 로 등록 → 학습되지 않음(고정 prior).           [P5]
    - 출력 β 는 s 에만 의존. s 가 학습되면 β 도 변하지만, '앵커 곡선 위'에
      갇혀 물리적으로 타당한 범위를 벗어나지 못한다.
    """
    # Solonenko & Mobley (2015) 실측 IOP β_D (R,G,B), 6 water types.
    # turbidity 축 = β_B (B채널 감쇠율) 오름차순: I<II<III<1C<5C<9C
    #   (Jerlov: Ocean I,II,III = 맑은 외양수 / Coastal 1C,5C,9C = 탁한 연안수)
    #
    # 주의(물리적 사실, 숨기지 않고 명시):
    #   β_B 는 이 축에서 단조 증가하지만 β_R·β_G 는 '비단조'.
    #   β_R: I(0.345) 최대 → III(0.135) 최소 → 9C(0.290) 재상승 (U자).
    #   이유: β_R 은 turbidity 가 아니라 '외양수 vs 연안수' 축을 반영
    #        (맑은 외양수도 red 를 강하게 흡수 → 파랗게 보임).
    #   → 1D 좌표 s 의 piecewise-linear 보간은 앵커점에서 정확, 사이만 선형근사.
    #     β_R U자를 정밀 표현하려면 2D 잠재좌표로 확장 가능(나중 정밀화). 현재 1D.
    #
    # 순서:   I      II     III    1C     5C     9C
    # β_B:  0.017  0.024  0.038  0.047  0.245  0.349   (단조 = s 축)
    DEFAULT_ANCHORS = torch.tensor([
        # βR,    βG,    βB
        [0.345, 0.073, 0.017],  # I    s=0.0  가장 맑음(외양수)
        [0.179, 0.082, 0.024],  # II
        [0.135, 0.089, 0.038],  # III  (β_R 최소점)
        [0.179, 0.082, 0.047],  # 1C
        [0.245, 0.156, 0.245],  # 5C
        [0.290, 0.199, 0.349],  # 9C   s=1.0  가장 탁함(연안수)
    ], dtype=torch.float32)

    def __init__(self, anchors: torch.Tensor = None):
        super().__init__()
        a = anchors if anchors is not None else self.DEFAULT_ANCHORS
        self.register_buffer("anchors", a.clone())      # [K,3] 고정
        self.K = a.shape[0]
        # 앵커의 s 좌표 = 정규화된 β_B (turbidity 정의 = β_B).
        # 등간격이 아니라 실제 β_B 비례 → s 가 실제 물리 감쇠율을 직접 반영.
        # (맑은 물 I,II,III,1C 는 s≈0~0.09 에 촘촘, 5C/9C 로 크게 점프 —
        #  실제 Solonenko&Mobley 분포 그대로.)
        betaB = a[:, 2]
        s_coord = (betaB - betaB.min()) / (betaB.max() - betaB.min() + 1e-9)
        self.register_buffer("anchor_s", s_coord.clone())

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        s : [B] 또는 [B,1]  in [0,1]
        return β : [B,3]  (β_R,β_G,β_B), 미분 가능(s 에 대해), 앵커는 고정.
        piecewise-linear monotone 보간.
        """
        s = s.view(-1).clamp(0, 1)                       # [B]
        # anchor_s 는 비등간격(β_B 비례). searchsorted 로 구간 탐색 후 선형 보간.
        a_s = self.anchor_s                              # [K], 단조 증가
        # 각 s 가 속한 구간 [lo,hi] 찾기
        hi = torch.searchsorted(a_s, s, right=True).clamp(1, self.K - 1)  # [B]
        lo = hi - 1
        s_lo = a_s[lo]; s_hi = a_s[hi]                    # [B]
        denom = (s_hi - s_lo).clamp_min(1e-9)
        w = ((s - s_lo) / denom).unsqueeze(1)            # [B,1] 보간 가중
        beta_lo = self.anchors[lo]                       # [B,3]
        beta_hi = self.anchors[hi]                       # [B,3]
        beta = (1 - w) * beta_lo + w * beta_hi           # [B,3]
        return beta


# =====================================================================
# WaterTypePredictor — g_φ  [SPEC 3장]
# =====================================================================
class WaterTypePredictor(nn.Module):
    """
    이미지 → s(turbidity 좌표). 두 모드:
      - Mode 1 (training-free): learn=False.
          이미지에서 near-field log(R/B) 통계로 s 를 '휴리스틱' 산출. 학습 없음.
      - Mode 2 (learned): learn=True.
          작은 CNN 이 s 를 예측. pseudo-label loss 로 방향 정렬(옵션).
    두 경우 모두 출력은 s∈[0,1] 스칼라(배치별).
    """
    def __init__(self, learn: bool = False):
        super().__init__()
        self.learn = learn
        if learn:
            # 아주 가벼운 회귀 헤드 (Mode 2). 저차원 좌표만 뽑으므로 작게.
            self.net = nn.Sequential(
                nn.Conv2d(3, 16, 3, 2, 1), nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(32, 1),
            )

    @staticmethod
    def _heuristic_s(img: torch.Tensor) -> torch.Tensor:
        """
        Mode 1: near-field(밝기 상위) log(R/B) 로 s 추정.
        탁할수록 R/B 가 커지는 경향 → s 를 그 방향으로 매핑.
        img: [B,3,H,W] (RGB 순 가정; 아래 PPCM 에서 채널순 관리)
        """
        B = img.shape[0]
        eps = 1e-4
        R = img[:, 0]; Bl = img[:, 2]
        lum = img.mean(1)                                    # [B,H,W]
        s_list = []
        for b in range(B):
            l = lum[b].flatten()
            thr = torch.quantile(l, 0.8)
            m = l >= thr
            r = R[b].flatten()[m].mean()
            bl = Bl[b].flatten()[m].mean()
            logRB = torch.log((r + eps) / (bl + eps))
            s_list.append(logRB)
        s = torch.stack(s_list)                              # [B]
        # logRB 범위를 [0,1] 로 정규화(대략적). 실제 데이터 분포로 보정 예정(미룸).
        s = torch.sigmoid(s)                                 # 단조 매핑
        return s

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if self.learn:
            s = torch.sigmoid(self.net(img)).view(-1)        # [B] in [0,1]
        else:
            with torch.no_grad():
                s = self._heuristic_s(img)
        return s


# =====================================================================
# Stage 1 — 채널 residual, 평균중심  [P1][N3]
# =====================================================================
class PPCMStage1(nn.Module):
    """
    r_c = exp(-β_D^c);  I'_c = I_c · (1 + α·(r_c - r̄))
    - residual + 평균중심. 곱셈 덮어쓰기/빼기 복원 아님.               [P1]
    - depth 사용 안 함 (water type만).                               [N3]
    - α 는 learnable 스칼라.
    """
    def __init__(self, alpha_init: float = 0.1):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, img: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """
        img : [B,3,H,W]  (채널순 = R,G,B)
        beta: [B,3]      (β_R,β_G,β_B)
        """
        r = torch.exp(-beta)                                 # [B,3] reliability
        r_bar = r.mean(dim=1, keepdim=True)                  # [B,1]
        gain = 1.0 + self.alpha * (r - r_bar)                # [B,3] 평균중심
        gain = gain.view(-1, 3, 1, 1)                        # broadcast
        return img * gain                                    # residual scale


# =====================================================================
# Stage 2 — 채널별 transmission, zero-init γ residual, centering
#           [P2][P3][P4][N4]
# =====================================================================
class PPCMStage2(nn.Module):
    """
    각 neck 스케일 N_l 에 대해:
      t_c = exp(-β_D^c · z)                 채널별 transmission (평균 안 냄) [P2]
      T   = [t_R, t_B]                       (채널축 유지; G 는 R/B 중간이라 생략 가능)
      T_l,z_l = downsample → centering       mean 빼기(=1 나눗셈 아님)       [P4]
      N'_l = N_l + γ_l · Conv([N_l,T_l,z_l]) zero-init γ residual           [P3]
    - 스케일마다 독립 Conv 와 독립 γ_l.
    - γ_l = 0 초기화 → 학습시작 = 순정.
    """
    def __init__(self, in_channels_per_scale, n_transmission=2,
                 depth_min=0.5, depth_max=10.0):
        """
        in_channels_per_scale: list[int], 각 neck 스케일 채널 수 (예: [256,512,512] 등)
        n_transmission: T 에 넣을 채널 수 (2 = [t_R,t_B], 3 = [t_R,t_G,t_B])
        depth_min/max: [0,1] 정규화 depth 를 미터 스케일로 환산하는 범위.
          z_m = depth_min + depth01 * (depth_max - depth_min)
          이유: [0,1] 그대로면 exp(-β·z) 의 β·z 가 ≤0.35 로 너무 작아
                transmission 이 0.7~1.0 에서만 미세 변동 → depth 정보 소실.
                미터 스케일(0.5~10)이면 β·z 가 0.17~3.45 로 물리적 감쇠를 만듦.
                (기존 dataset 코드의 z=0.5+depth*9.5 와 동일)
        """
        super().__init__()
        self.n_t = n_transmission
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.convs = nn.ModuleList()
        self.gammas = nn.ParameterList()
        for ch in in_channels_per_scale:
            # 입력: N_l(ch) + T(n_t) + z(1)
            self.convs.append(nn.Conv2d(ch + n_transmission + 1, ch, kernel_size=1))
            self.gammas.append(nn.Parameter(torch.zeros(1)))   # zero-init [P3]

    def _transmission(self, z, beta):
        """
        z   : [B,1,H,W] in [0,1]  (0=가까움, 1=멀다)
        beta: [B,3]
        return T: [B,n_t,H,W]  (채널별 t_c = exp(-β_c · z_meters))
        """
        # [0,1] → 미터 스케일 (물리적 감쇠를 만들기 위해)
        z_m = self.depth_min + z * (self.depth_max - self.depth_min)
        # 채널 선택: n_t=2 → R,B ; n_t=3 → R,G,B
        idx = [0, 2] if self.n_t == 2 else [0, 1, 2]
        ts = []
        for c in idx:
            bc = beta[:, c].view(-1, 1, 1, 1)                # [B,1,1,1]
            ts.append(torch.exp(-bc * z_m))                  # [B,1,H,W]
        return torch.cat(ts, dim=1)                          # [B,n_t,H,W]

    def forward(self, feats, z, beta):
        """
        feats: list of [B,C_l,H_l,W_l]  (neck 출력 N3,N4,N5)
        z    : [B,1,H,W]  원해상 depth
        beta : [B,3]
        return list of N'_l (같은 shape)
        """
        T_full = self._transmission(z, beta)                 # [B,n_t,H,W]
        out = []
        for l, N in enumerate(feats):
            Hl, Wl = N.shape[-2:]
            T_l = F.interpolate(T_full, size=(Hl, Wl), mode="bilinear", align_corners=False)
            z_l = F.interpolate(z,      size=(Hl, Wl), mode="bilinear", align_corners=False)
            # centering: mean 빼기 (detach 로 scale 은 γ 에 위임)          [P4]
            T_l = T_l - T_l.mean(dim=(2, 3), keepdim=True).detach()
            z_l = z_l - z_l.mean(dim=(2, 3), keepdim=True).detach()
            mod = self.convs[l](torch.cat([N, T_l, z_l], dim=1))
            out.append(N + self.gammas[l] * mod)             # zero-init residual [P3]
        return out


# =====================================================================
# PPCM — 토글 컨테이너  [P7]
# =====================================================================
class PPCM(nn.Module):
    """
    stage1_on / stage2_on 토글로 4개 구성 제어.  [P7]
      baseline: (F,F)  stage1: (T,F)  stage2: (F,T)  full: (T,T)
    forward-hook 부착은 attach_* 헬퍼로 (YOLOv8 무수정).  [P6]

    채널순 주의: ultralytics 는 RGB 입력. 내부 R,G,B 인덱스 = 0,1,2 가정.
    """
    def __init__(self,
                 stage1_on: bool,
                 stage2_on: bool,
                 neck_channels=None,        # stage2_on 일 때 필수: list[int]
                 learn_gphi: bool = False,  # Mode2 스위치
                 n_transmission: int = 2,
                 anchors: torch.Tensor = None):
        super().__init__()
        self.stage1_on = stage1_on
        self.stage2_on = stage2_on

        self.phys = PhysInterp(anchors)
        self.gphi = WaterTypePredictor(learn=learn_gphi)
        self.stage1 = PPCMStage1() if stage1_on else None
        if stage2_on:
            assert neck_channels is not None, "stage2_on 이면 neck_channels 필요"
            self.stage2 = PPCMStage2(neck_channels, n_transmission=n_transmission)
        else:
            self.stage2 = None

        # 현재 배치의 z,beta 를 hook 에 전달하기 위한 캐시
        self._z = None
        self._beta = None

    # ---- 배치 준비: s→β, depth 확보 ----
    def prepare(self, img: torch.Tensor, depth: torch.Tensor):
        """
        매 배치 forward 전에 호출.
        img  : [B,3,H,W] RGB [0,1]
        depth: [B,1,H,W] [0,1]  (WaterMono 등에서 미리 계산)
        """
        s = self.gphi(img)                 # [B]
        beta = self.phys(s)                # [B,3]  (β 는 앵커 고정, s 통해서만 변화)
        self._beta = beta
        self._z = depth
        return beta

    # ---- Stage1: 입력 전처리 (backbone 진입 전) ----
    def apply_stage1(self, img: torch.Tensor) -> torch.Tensor:
        if not self.stage1_on:
            return img                     # 완전 무동작 (identity)  [P7]
        return self.stage1(img, self._beta)

    # ---- Stage2: neck 출력 hook 콜백 ----
    def apply_stage2(self, feats):
        if not self.stage2_on:
            return feats                   # 완전 무동작  [P7]
        return self.stage2(feats, self._z, self._beta)


# =====================================================================
# forward-hook 부착 헬퍼 (ultralytics 연결부)  [P6][N5]
# =====================================================================
def attach_ppcm_hooks(yolo_model, ppcm: PPCM, detect_index=None):
    """
    ultralytics YOLO 모델에 PPCM 을 forward-hook 으로 부착.  [P6][N5]
    YOLOv8 파라미터/구조는 일절 수정하지 않고 hook 만 건다.

    Stage1 (입력 전처리):
        model.model[0] (backbone stem) 의 forward-PRE-hook 으로
        입력 이미지 I -> I' 치환. backbone 진입 전에만 적용.

    Stage2 (Detect 입력 치환):
        ★ 개별 neck 레이어(YOLOv8m 의 15,18,21)에 hook 을 걸면 안 된다.
          15,18 의 출력은 Detect 로 갈 뿐 아니라 하류 neck 레이어
          (16->17->18, 19->20->21)로도 분기되므로, 거기서 치환하면
          neck 내부 계산이 오염된다 (P6 위반).
        ★ 대신 Detect 레이어의 forward-PRE-hook 을 쓴다.
          Detect.forward 의 입력은 [N3,N4,N5] 리스트이며, 여기서 치환하면
          neck 내부는 순정 그대로, 오직 Detect 입력만 PPCM 을 통과한다.

    detect_index: Detect 레이어 인덱스. None 이면 자동 탐색.
      (YOLOv8n/s/m/l/x 모두 마지막 레이어가 Detect. YOLOv8m = 22)

    반환: hook handle 리스트 (제거용)
    """
    handles = []
    model_seq = yolo_model.model  # ultralytics DetectionModel.model (nn.Sequential 류)

    # --- Stage 1: backbone stem pre-hook ---
    def stage1_pre_hook(module, args):
        x = args[0]                       # [B,3,H,W]
        x = ppcm.apply_stage1(x)
        return (x,) + tuple(args[1:])

    handles.append(model_seq[0].register_forward_pre_hook(stage1_pre_hook))

    # --- Stage 2: Detect pre-hook ---
    if detect_index is None:
        detect_index = len(model_seq) - 1
        for i, m in enumerate(model_seq):
            if type(m).__name__ == "Detect":
                detect_index = i
                break

    def detect_pre_hook(module, args):
        # ultralytics Detect.forward(self, x) 에서 x = [N3,N4,N5] (list)
        feats = args[0]
        if not isinstance(feats, (list, tuple)):
            return args                   # 예상과 다르면 무동작(안전)
        mod = ppcm.apply_stage2(list(feats))
        return (mod,) + tuple(args[1:])

    handles.append(model_seq[detect_index].register_forward_pre_hook(detect_pre_hook))

    return handles