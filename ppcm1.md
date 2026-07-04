# PPCM 파이프라인 확정 명세 (SPEC) — 구현의 기준점

이 문서는 지금까지 합의된 파이프라인을 **정확히** 고정한다.
구현(코드)은 이 명세와 1:1 대응해야 하며, 여기 없는 것은 넣지 않고
여기 있는 것은 빼지 않는다.

================================================================
0. 전체 데이터 흐름 (확정 다이어그램)
================================================================

Raw Image (BGR, [0,1])
  │
  ├─→ [연속 water-type 예측기 g_φ(I)]  → s (저차원 turbidity 좌표)
  │        └─ 물리 보간(PhysInterp) → β_D^c = (β_R, β_G, β_B)   [채널별 감쇠계수]
  │
  ├─→ [WaterMono depth]  → z(x)  (relative, [0,1])
  │
  ↓
[Stage 1: 채널 신뢰도 주입 — water type만 사용, depth 사용 안 함]
  r_c = exp(-β_D^c)                    # 채널별 reliability
  r̄  = mean_c(r_c)                     # 3채널 평균
  I'_c = I_c · (1 + α·(r_c - r̄))       # residual, 평균중심. 곱셈-덮어쓰기 아님
  # → 신뢰 채널 상대 강조 / 비신뢰 채널 상대 억제. 정보 삭제 없음.
  ↓
[YOLOv8 Backbone + C2f Neck : 순정, pretrained. 절대 수정 안 함]
  → N3, N4, N5   (neck 출력 3개 스케일)
  ↓
[Stage 2: 공간 신뢰도 주입 — water type + depth 사용]
  z_m(x) = 0.5 + z(x) · 9.5            # [0,1] depth → 미터 스케일 [0.5,10]
                                        #   (β·z 가 물리적 감쇠 범위 갖도록; 필수)
  t_c(x) = exp(-β_D^c · z_m(x))        # 채널별 transmission. RGB 평균 내지 않음
  T(x)   = [t_R, t_B]  또는 [t_R,t_G,t_B]  (confidence map, 채널축 유지)
  각 스케일 l ∈ {N3,N4,N5} 에 대해:
    T_l  = downsample(T,  size(N_l))
    z_l  = downsample(z,  size(N_l))
    T_l  = T_l  - detach(mean(T_l))     # centering. mean=1 나눗셈 아님
    z_l  = z_l  - detach(mean(z_l))
    N'_l = N_l + γ_l · Conv([N_l, T_l, z_l])   # residual, zero-init γ_l
  # γ_l 은 스케일별 스칼라, 0으로 초기화 → 학습시작=순정 YOLOv8
  ↓
[YOLOv8 Detect Head : 순정, pretrained. 절대 수정 안 함]
  → boxes, scores, labels


================================================================
1. 토글 정의 (4개 구성)
================================================================
config 로 stage1_on / stage2_on 두 불리언을 켜고 끈다.

  baseline : stage1_on=False, stage2_on=False   → 순정 YOLOv8 그대로
  stage1   : stage1_on=True,  stage2_on=False
  stage2   : stage1_on=False, stage2_on=True
  full     : stage1_on=True,  stage2_on=True

토글이 꺼진 stage 는 '존재하지 않는 것과 완전히 동일'해야 한다
(no-op 이 아니라, hook 자체를 걸지 않거나 identity 로 통과).


================================================================
2. 삽입 방식 = forward hook (yaml 수정 안 함)
================================================================
- 순정 ultralytics YOLOv8 모델 로드.
- Stage 1: 입력 전처리 단계에서 적용 (backbone 진입 전).
    → dataset/collate 또는 model.forward 진입부에서 I → I' 변환.
- Stage 2: neck 출력 3개 텐서에 forward hook.
    → hook 이 N3,N4,N5 를 받아 N'3,N'4,N'5 로 치환 후 반환.
- backbone/neck/head 의 파라미터·구조는 일절 건드리지 않음.


================================================================
3. 학습 대상 파라미터 (무엇이 learnable 인가)
================================================================
확정 사항:
  - β_D 자체: 고정 prior (Solonenko & Mobley 앵커). gradient 안 받음.
  - g_φ (water-type 예측기): learnable.  → s 예측 → 물리보간으로 β_D^c.
  - α (Stage1 residual 강도): learnable 스칼라.
  - γ_l (Stage2 스케일별 gate): learnable, zero-init.
  - Stage2 Conv: learnable.
  - YOLOv8 backbone/neck/head: 기존대로 (pretrained fine-tune).

training-free 포지션(Mode 1)을 위해:
  - g_φ 없이 휴리스틱/이산으로 s 를 줄 수도 있어야 함 → g_φ 를 끌 수 있는 스위치.
  (Mode 2 = g_φ 학습. 이건 옵션으로 분리, 기본 파이프라인엔 스위치만 남김)


================================================================
4. pseudo-label supervision (Stage1 학습 보조)
================================================================
- near-field 채널비율 log(I_R/I_B) 기반 rank/monotonicity loss.
- g_φ 예측 β 스펙트럼 기울기와 관측 기울기의 '방향'만 정렬 (절대값 회귀 아님).
- 이건 Stage1(g_φ)이 켜지고 Mode 2 일 때만 활성. baseline/stage2 엔 없음.
- 검증 4 (MODE B, 실제 depth) 통과 후 최종 확정.  ← 지금은 스위치만.


================================================================
5. 절대 규칙 (빠지지도 더해지지도 않게)
================================================================
[포함되어야 하는 것 — 누락 시 실패]
  P1. Stage1 = residual + 평균중심 (곱셈 덮어쓰기 금지)
  P2. Stage2 = 채널별 transmission (RGB 평균 금지)
  P3. Stage2 = zero-init γ residual (곱셈 게이팅 금지)
  P4. Stage2 = centering (mean=1 나눗셈 금지)
  P4b. Stage2 = z 를 미터 스케일(0.5~10)로 변환 후 transmission 계산
       ([0,1] 그대로 넣으면 감쇠가 미미해 depth 정보 소실)
  P5. β_D = 고정 prior (자유 학습 금지)
  P6. YOLOv8 backbone/neck/head 무수정
  P7. 4개 토글이 정확히 위 정의대로 동작

[더해지면 안 되는 것 — 있으면 실패]
  N1. restoration/dehazing 류 (backscatter 명시 복원 등) — 우리 목적 아님
  N2. depth-aware scale routing — 보류 항목, 기본 파이프라인에 넣지 않음
  N3. Stage1 에 depth 사용 금지 (Stage1 은 water type만)
  N4. Stage2 를 pixel space 에 적용 금지 (Stage2 는 neck 출력에만)
  N5. 임의의 attention/정규화 레이어 추가 금지 (명세에 있는 Conv 만)


================================================================
6. 지금 단계에서 '스위치만 두고 미완'인 것 (합의된 미룸)
================================================================
  - g_φ 실제 구조 (Mode 2): 스위치 O, 내부는 최소 구현 후 나중 정밀화
  - pseudo-label loss: 스위치 O, 검증4(MODE B) 후 확정
  - β_D 물리보간(PhysInterp) 곡선: 6점 앵커, 나중 정밀화
  - RUOD clear/turbid 분할: log(R/B) 기반, 학습 직전 수행
  이들은 '기본 파이프라인(baseline/stage1/stage2/full 돌리기)'에는
  영향 주지 않도록, 꺼진 상태를 기본값으로 둔다.