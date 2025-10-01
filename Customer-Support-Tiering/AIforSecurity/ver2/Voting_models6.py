#모델의 종류를 늘린 후, 가장 높은 정확도 모델 선택 
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    StackingClassifier
)
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from category_encoders import TargetEncoder
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 데이터 로드
try:
    train = pd.read_csv('./train.csv')
    test = pd.read_csv('./test.csv')
    sample_submission = pd.read_csv('./sample_submission.csv')
except FileNotFoundError:
    print("파일을 찾을 수 없습니다. ./train.csv, ./test.csv, ./sample_submission.csv 파일을 확인해주세요.")
    exit()

# 데이터 전처리
X = train.drop(['ID', 'attack_type'], axis=1)
y = train['attack_type']
test_df = test.drop(['ID'], axis=1)

# Target Encoding
category_features = ['ip_src', 'ip_dst', 'protocol']
for col in category_features:
    enc = TargetEncoder()
    X[col] = enc.fit_transform(X[col].fillna('Missing'), y)
    test_df[col] = enc.transform(test_df[col].fillna('Missing'))

# 수치형 변수 결측값 처리
numeric_cols = ['port_src', 'port_dst', 'duration', 'pkt_count_fwd', 'pkt_count_bwd',
                'rate_fwd_pkts', 'rate_bwd_pkts', 'rate_fwd_bytes', 'rate_bwd_bytes',
                'payload_fwd_mean', 'payload_bwd_mean', 'tcp_win_fwd_init', 'tcp_win_bwd_init',
                'tcp_syn_count', 'tcp_psh_count', 'tcp_rst_count', 'iat_avg_packets']

X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
test_df[numeric_cols] = test_df[numeric_cols].fillna(X[numeric_cols].mean())

# Target 라벨 인코딩
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 데이터 분할
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 모델 정의 (디폴트 파라미터 사용)
base_models = {
    "Random Forest": RandomForestClassifier(random_state=42, n_jobs=-1),
    "Extra Trees": ExtraTreesClassifier(random_state=42, n_jobs=-1),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    "KNN": KNeighborsClassifier(),
    "LightGBM": lgb.LGBMClassifier(objective='multiclass',num_class=3,
                                   random_state=42,n_jobs=-1,min_child_samples=5,  # 기본값, 필요시 더 줄임
                                   min_gain_to_split=0,  # 분할 최소 이득을 낮춤
                                   max_depth=-1          # 트리 깊이 제한 없음
                                   )

}

# 모델 훈련 및 평가
best_score = 0
best_model = None
best_model_name = ""
estimators_list = []

print("--- 개별 모델 훈련 및 평가 ---")
for name, model in base_models.items():
    print(f"\n▶ 모델 훈련: {name}")
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    score = f1_score(y_val, y_val_pred, average='macro')
    print(f"{name} Macro F1 Score: {score:.4f}")

    estimators_list.append((name, model))

    if score > best_score:
        best_score = score
        best_model = model
        best_model_name = name

# Voting & Stacking 앙상블 정의 (기본 모델 기반)
voting_ensemble = VotingClassifier(
    estimators=estimators_list,
    voting='soft',
    n_jobs=-1
)

stacking_ensemble = StackingClassifier(
    estimators=estimators_list,
    final_estimator=LogisticRegression(max_iter=1000),
    passthrough=True,
    n_jobs=-1
)

# 앙상블 모델 훈련 및 평가
ensemble_models = {
    "Voting Ensemble": voting_ensemble,
    "Stacking Ensemble": stacking_ensemble
}

print("\n--- 앙상블 모델 훈련 및 평가 ---")
for name, model in ensemble_models.items():
    print(f"\n▶ 모델 훈련: {name}")
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    score = f1_score(y_val, y_val_pred, average='macro')
    print(f"{name} Macro F1 Score: {score:.4f}")

    if score > best_score:
        best_score = score
        best_model = model
        best_model_name = name

print(f"\n=========================================")
print(f" 최종 최적 모델: {best_model_name} (Macro F1: {best_score:.4f})")
print(f"=========================================")

# 최종 예측 및 제출
test_pred = best_model.predict(test_df)
test_pred_labels = label_encoder.inverse_transform(test_pred)

sample_submission['attack_type'] = test_pred_labels
sample_submission.to_csv('./best_model_submission.csv', index=False)
