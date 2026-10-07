파일 전체 구조
```
KICS2026_2/
|-- team_env.py
|-- models/
|   |-- shared_actor_critic.py
|   `-- trajectory_encoder.py
|-- rewards/self_reference_reward.py
|-- algorithms/
|   |-- ppo.py
|   `-- srpo.py
|-- collect_trajectories.py       # canonical
|-- collect_rollouts.py           # compatibility re-export only
|-- train_ppo.py
|-- train_srpo.py
|-- pretrain_encoder.py
|-- evaluate.py
|-- training_utils.py
|-- environment.yml
`-- tests/
```

# team_env.py
```
env = OvercookedTeamEnv(
    layout_name="cramped_room",
    horizon=400,
    target_deliveries=1,
    reward_mode="shaped",  # "sparse", "shaped", "binary" 중 하나
    shaping_coef=0.1,
)
```
한 스텝의 플레이를 만들고, `reward_mode`에 따라 한 스텝에 대한 보상을 반환 
`step`이란 최대 400턴에 해당하는 플레이어 행동 집합, 매 턴 마다의 관측을 통해 배달 수로 보상을 계산한다. 

# shared_actor_critic.py
**model 인스턴스는 train_ppo.py 또는 train_srpo.py에서 생성한다.**
self.actor : 각 플레이어의 관측 상태를 입력받아 다음 행동 결정을 위한 logit 출력
self.critic : 두 플레이어의 관측을 모두 입력받아 현재 상태에 대한 value(예측되는 누적 보상) 출력
두 모델은 모두 MLP구조로, 선형 레이어`nn.linear`와 활성화함수 `nn.Tanh`가 반복되다가, 마지막 행동 차원`action.dim`크기의 출력층으로 끝나는 구조이다.( `hidden_sizes=(256, 128)` 입력 -> 256 -> Tanh -> 128 -> Tanh -> Action Dim) 
critic model(value model)의 경우, 보상을 출력하므로, 마지막 출력층(Action Dim)의 크기가 1이다. 입력크기도 두 플레이어의 관측을 joint한만큼 더 크다. 
PPO에서 actor, critic모두 사용되고, SRPO에서는 actor만 사용된다. 

# collect_trajectories.py
현재 정책으로 환경을 실행해 on-policy 데이터를 저장한다. loss, optimizer는 다루지 않는다. 
한 episode의 모든 관측, 행동, 보상, 종류 여부, 배달 수를 저장한다. 
observation[t] → action[t] → reward[t] → observation[t+1] 정렬을 유지

# algorithm/ppo.py
환경 reward와 centralized critic을 사용해 GAE·return을 만들고, clipped policy loss·value loss·entropy로 actor와
critic을 업데이트한다.

# train_ppo.py
PPO 실험의 CLI와 전체 실행 순서를 관리하고, turn마다 JSONL·trajectory summary·checkpoint를 저장한다
1 PPO turn = episodes-per-turn개의 episode 수집 + 한 번의 PPO update + 로그 한 줄이다.
수집 → GAE → update → log → checkpoint 반복

# models/trajectory_encoder.py
관측 시계열을 하나의 latent vector로 압축하는 GRU encoder와, encoder 사전학습용 action-conditioned next-state predictor를 정의한다.
GRU의 최종 은닉 상태를 최종 임베딩 차원으로 변환하여 출력한다. 
