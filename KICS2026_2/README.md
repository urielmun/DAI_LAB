# KICS2026_2: Overcooked PPO와 SRPO-inspired 협력 학습

두 에이전트가 함께 요리를 배달하는 Overcooked-AI 환경에서 다음을 비교하기 위한 기준 구현입니다.

- PPO: 환경의 timestep 보상과 centralized critic을 사용하는 기준선
- PPO-Binary: 성공 여부만 보상으로 쓰는 검증 가능 terminal reward 기준선
- SRPO-inspired: 현재 그룹의 성공 궤적을 참조해 실패 궤적 점수를 만들고 group-relative advantage로 actor를 업데이트

SRPO 코드는 원 논문의 VLA 구현을 그대로 복제한 것이 아닙니다. 영상 기반 V-JEPA2 대신 symbolic observation을 처리하는 GRU를 사용하는 Overcooked 변형입니다.

## 파일명 기준

정식 수집 파일은 `collect_trajectories.py`입니다. 모든 trainer가 여기서 `collect_trajectory_group()`을 가져옵니다. `collect_rollouts.py`는 이전 이름으로 작성한 외부 코드가 깨지지 않도록 둔 재수출 호환 파일이며 실제 로직은 없습니다.

## 설치

```bash
cd ~/KICS2026_2
conda env create -f environment.yml
conda activate overcooked-sr
which python
python -m pip check
```

`which python` 결과가 `/usr/bin/python3`이면 Conda 환경이 활성화되지 않은 것입니다.

CUDA 확인:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

CUDA가 없는 컴퓨터에서는 `environment.yml`의 `pytorch-cuda` 항목을 제거하고 conda-forge의 CPU PyTorch를 설치하거나 실행 시 `--device cpu`를 사용합니다.

## 검증

```bash
python -m compileall .
python -m pytest -q -m "not integration"
python -m pytest -q -m integration
```

## PPO 10턴 smoke test

```bash
python train_ppo.py \
  --turns 10 \
  --episodes-per-turn 1 \
  --layout-name cramped_room \
  --horizon 400 \
  --reward-mode shaped \
  --device auto \
  --seed 42 \
  --output-dir runs/smoke/ppo_10
```

로그는 `runs/smoke/ppo_10/training_log.jsonl`, 모델은 `runs/smoke/ppo_10/checkpoints/latest.pt`에 저장됩니다.

## 연구용 SRPO 준비 순서

1. 성공 궤적을 생성할 수 있는 PPO warm-start를 학습합니다.

```bash
python train_ppo.py \
  --turns 300 \
  --reward-mode shaped \
  --device auto \
  --output-dir runs/warmstart_ppo
```

2. 별도 데이터로 GRU encoder를 다음 상태 예측 과제로 사전학습합니다.

```bash
python pretrain_encoder.py \
  --turns 50 \
  --group-size 8 \
  --policy-checkpoint runs/warmstart_ppo/checkpoints/latest.pt \
  --device auto \
  --output-dir runs/encoder_pretrain
```

3. 두 체크포인트를 고정 출발점으로 SRPO 10턴을 실행합니다.

```bash
python train_srpo.py \
  --turns 10 \
  --group-size 8 \
  --layout-name cramped_room \
  --horizon 400 \
  --target-deliveries 1 \
  --checkpoint runs/warmstart_ppo/checkpoints/latest.pt \
  --encoder-checkpoint runs/encoder_pretrain/checkpoints/latest.pt \
  --device auto \
  --seed 42 \
  --output-dir runs/smoke/srpo_10
```

encoder 없이 코드 경로만 검사하려면 `--allow-untrained-encoder`를 사용할 수 있습니다. 이 경우 latent distance는 연구적 의미가 없으며 성능 결과로 사용하면 안 됩니다.

## 평가

```bash
python evaluate.py \
  --checkpoint runs/smoke/ppo_10/checkpoints/latest.pt \
  --episodes 100 \
  --device auto \
  --output-dir runs/eval/ppo_10
```

주요 지표는 정확도가 아니라 성공률, 평균 배달 횟수, 동일한 환경 step에서의 성공률입니다.

## 알고리즘 경계

| 항목 | PPO | SRPO-inspired |
|---|---|---|
| 환경 | 공통 | 공통 |
| actor | 공유 actor 사용 | 같은 actor 재사용 |
| critic | 학습 | 사용·학습하지 않음 |
| advantage | GAE | 그룹 점수 정규화 |
| encoder | 없음 | frozen trajectory encoder |
| reference policy | 없음 | SRPO 시작 정책을 고정 |
| optimizer 대상 | actor + critic | actor만 |

`old policy`는 수집 당시 저장한 공동 `old_log_probs`로 나타내며 그룹마다 바뀝니다. `reference policy`는 SRPO 시작점의 모델 복사본이며 전체 실행 동안 바뀌지 않습니다.

## 현재 구현과 남은 연구 검증

구현되어 있는 것:

- Gymnasium 2-agent wrapper와 실제 delivery event 집계
- shared actor, centralized critic, 공동 log probability
- PPO GAE, clipping, value loss, entropy, JSONL, checkpoint
- variable-length trajectory group padding과 mask
- GRU encoder 구조와 다음 상태 예측 사전학습 스크립트
- 성공 embedding DBSCAN, fallback, 실패 거리 점수
- group-relative advantage, fixed-reference KL, actor-only SRPO update
- 단위 테스트와 실제 Overcooked 통합 테스트

아직 연구 결과로 검증되지 않은 것:

- 사전학습된 encoder checkpoint 자체와 latent-progress 상관 분석
- 원 논문의 V-JEPA2 표현을 사용한 완전 재현
- timestep별 progress reward; 현재는 trajectory score 하나를 모든 timestep에 반복
- 동일 warm-start·동일 environment-step 예산의 PPO-Binary/SRPO 다중 seed 비교
- DBSCAN과 KL·group size hyperparameter 탐색
- 다른 layout 일반화, 협력 품질 지표, 신뢰구간과 통계 검정

PPO의 1 turn은 `episodes-per-turn`개의 episode를 수집한 뒤 한 번 업데이트하는 단위입니다. SRPO의 1 turn은 `group-size`개의 episode를 수집한 뒤 한 번 업데이트하는 단위입니다. 따라서 두 방법의 turn 수를 직접 비교하지 말고 `global_env_steps`를 맞춰야 합니다.

## 출처

- Overcooked-AI: https://github.com/HumanCompatibleAI/overcooked_ai (검증 commit `739950a079cdaed5a44fcc662efc40244c205d06`)
- PPO: https://arxiv.org/abs/1707.06347
- SRPO: https://arxiv.org/abs/2511.15605
