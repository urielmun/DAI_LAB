# 🧠 AI Cross-lab Tech Seminar

충북대학교 AI Cross-lab에서 진행한 테크 세미나 발표 자료 및 논문 리뷰 아카이브입니다. 대형 언어 모델(LLM)과 비전-언어-행동 모델(VLA) 분야의 최신 인공지능 연구 동향과 핵심 알고리즘을 분석합니다.

---

## 📁 Seminar Archives

### 1. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
* **발표 일자**: 2026년 04월 10일
* **발표자**: 문서영 (DAI LAB)
* **주요 내용**:
  * **Pure RL 가능성 증명**: DeepSeek-R1-Zero를 통한 순수 강화학습 기반 추론 능력 유도 확인[cite: 4]
  * **GRPO 알고리즘**: 그룹 상대 정책 최적화(Group Relative Policy Optimization) 도입[cite: 4, 7]
  * **Multi-stage Training Pipeline**: 콜드 스타트 및 다단계 학습 파이프라인 구축[cite: 4, 9]
  * 작은 모델에서도 고성능 추론 능력을 확보할 수 있음을 입증[cite: 4]

### 2. SRPO: Self-Referential Policy Optimization for Vision-Language-Action Models
* **발표 일자**: 2026년 07월 24일[cite: 14]
* **발표자**: 문서영 (Dai Lab)[cite: 14]
* **주요 내용**:
  * **자기 참조 정책 최적화 (SRPO)**: 비전-언어-행동(VLA) 모델을 위한 온라인 포스트 트레이닝 효과 입증[cite: 17]
  * **보상 설계 (Reward Design)**: 성공 참조 궤적과 월드 모델 잠재 공간(Latent World Representation) 기반의 조밀한 과정 보상(Dense Reward) 부여[cite: 16, 18, 20]
  * 소스 데이터 및 시각 입력만으로 SOTA 성능 달성 및 One-Shot SFT의 다양성 한계 극복[cite: 17]

### 3. Direct Preference Optimization: Your Language Model is Secretly a Reward Model (DPO)
* **발표 일자**: 2026년 09월 21일[cite: 25]
* **발표자**: 문서영 (Dai Lab)[cite: 25]
* **주요 내용**:
  * **DPO 제안**: 별도의 보상 모델 학습과 PPO 방식 대신, 인간 선호 데이터로 언어 모델을 직접 최적화하는 단순한 분류 목적 함수 제안[cite: 28, 30]
  * **Implicit Reward Model**: 언어 모델 자체가 암묵적인 보상 모델 역할을 수행함을 수학적으로 증명[cite: 28, 33]
  * 샘플링 온도 변화에 대한 강건성 입증 및 PPO 대비 우수한 성능 확인[cite: 37]

---

## 🛠️ Repository Structure

```text
├── seminar26.04.10.pdf    # DeepSeek-R1 논문 리뷰 발표자료
├── seminar26.07.24.pdf    # SRPO 논문 리뷰 발표자료
├── seminar26.09.21.pdf    # DPO 논문 리뷰 발표자료
└── README.md
