# AIforSecurity
---
사이버 공격 유형 예측 해커톤: 트래픽 속 위협을 식별하라!

# 인공지능 모델 선택
1. Task Type:

분류(Classification), 20개의 특성을 통해 12개의 결과 중 하나로 분류
{'GoldenEye', 'Slowloris', 'Benign', 'Web_Brute_Force', 'DDoS', 'SSH_Brute_Force', 'Botnet', 'Slow_HTTP', 'Web_XSS', 'FTP_Brute_Force', 'Port_Scanning', 'Hulk'}

1. 데이터 샘플 수:

3000개( 소량 데이터)→ 단순모델, 선형,로지스틱 회귀

2. 특성 수, 차원:

20개 특성(네트워크 트래픽)→ attack_type 예측

- 샘플 수가 아주 적어서 n<20n<20n<20 정도라면, 비록 피처가 20개여도 p≥np\ge np≥n 상황이 돼 모델링이 어렵고 “고차원 문제”로 간주될 수 있습니다.
- 반대로 샘플이 많아서 n≫20n\gg 20n≫20 이라면 차원이 낮은 편입니다.

차원 축소(PCA)나 정규화, 피처 선택(SelectKBest, RFE) 등을 적용할 수 있는 모델인지 점검

3. 해석 가능성(Interpretability):

내부 동작 이해 가능성

4. 학습, 추론 비용:

GPU·메모리·저장공간 제약을 고려해 모델 크기와 복잡도를 결정

5. 성능요구도(정확도 vs 속도):

평가 산식 : macro f1-score:

F1-score (클래스별): Precision과 Recall의 조화평균

Macro F1-score: 전체 C개의 클래스 각각의 F1 단순 평균낸 값

정확도>속도

#최적화 과정
1-1. 앙상블 모델의 적용
bagging→ RandomForestClassifier
boosting→XGBClassifier
softVoting→ VotingClassifier
세 개 모델의 train.csv 상의 F1-socre을 비교하여 정확도가 가장 높은 모델을 선택
최적 모델: XGBoost (Macro F1: 0.8562)
1-2. LabelEncoding→TargetEncoding
XGBoost (Macro F1: 0.8489)

2. 모델의 종류를 늘린 후 가장 높은 정확도 모델 선택
- -- 개별 모델 훈련 및 평가 ---
▶ 모델 훈련: Random Forest Random Forest Macro F1 Score: 0.8084

▶ 모델 훈련: Extra Trees Extra Trees Macro F1 Score: 0.7942

▶ 모델 훈련: AdaBoost AdaBoost Macro F1 Score: 0.2450

▶ 모델 훈련: XGBoost XGBoost Macro F1 Score: 0.8485

▶ 모델 훈련: KNN KNN Macro F1 Score: 0.4470

▶ 모델 훈련: LightGBM LightGBM Macro F1 Score: 0.8379

- -- 앙상블 모델 훈련 및 평가 ---
-
▶ 모델 훈련: Voting Ensemble

[LightGBM] [Warning] min_gain_to_split is set=0, min_split_gain=0.0 will be ignored. Current value: min_gain_to_split=0

Voting Ensemble Macro F1 Score: 0.8363

▶ 모델 훈련: Stacking Ensemble

[LightGBM] [Warning] min_gain_to_split is set=0, min_split_gain=0.0 will be ignored. Current value: min_gain_to_split=0

Stacking Ensemble Macro F1 Score: 0.0815

=========================================

최종 최적 모델: XGBoost (Macro F1: 0.8485)

2-2. 파라미터값 최적화(그리드 서치 이용)

각 모델의 기본값 대신 가장 높은 정확도를 기록하는 파라미터 값을 선택하여 업데이트

최종 최적 모델: XGBoost (Macro F1: 0.8489)

#최종 리서치

해당 대회는 인공지능 모델을 사용하여 20개의 특성을 분석하여 사이버 공격 유형 12개 중 하나로 분류한다.

따라서 분류 모델의 적용이 유리하다. decisionTree base/Ensemble

- Random Forest Macro F1 Score: 0.8084
- Voting Ensemble Macro F1 Score: 0.8363
- LightGBM Macro F1 Score: 0.8379
- XGBoost Macro F1 Score: 0.8485

