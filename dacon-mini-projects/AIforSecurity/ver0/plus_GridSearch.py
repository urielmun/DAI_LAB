#파라미터값 최적화
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    StackingClassifier
)
import lightgbm as lgb
from sklearn.datasets import load_breast_cancer

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
import numpy as np
!pip install category_encoders

train = pd.read_csv('./train.csv') #  읽어서 저장
test = pd.read_csv('./test.csv')
sample_submission = pd.read_csv('./sample_submission.csv')

X = train.drop(['ID','attack_type'], axis=1)
y = train['attack_type']

test = test.drop(['ID'], axis=1)

# 범주형 데이터 인코딩
category_features = ['ip_src', 'ip_dst', 'protocol']
encoders = {}
from category_encoders import TargetEncoder
def safe_transform(encoder, series):
    """Train에 없는 값을 -1로 처리"""
    known_classes = set(encoder.classes_)
    print(encoder.classes_)
    print(known_classes)
    return series.apply(lambda x: encoder.transform([x])[0] if x in known_classes else -1)
'''
for col in category_features:
    enc = LabelEncoder()
    X[col] = enc.fit_transform(X[col].fillna('Missing'))
    test[col] = safe_transform(enc, test[col].fillna('Missing'))
    encoders[col] = enc
    '''
for col in category_features:
    #enc = LabelEncoder()
    enc = TargetEncoder()

    X[col] = enc.fit_transform(X[col].fillna('Missing'),y)
    test[col] = enc.transform(test[col].fillna('Missing'))
    encoders[col] = enc

# 수치형 변수 결측값 처리
numeric_cols = ['port_src','port_dst','duration','pkt_count_fwd','pkt_count_bwd',
                'rate_fwd_pkts', 'rate_bwd_pkts', 'rate_fwd_bytes', 'rate_bwd_bytes',
                'payload_fwd_mean', 'payload_bwd_mean', 'tcp_win_fwd_init', 'tcp_win_bwd_init',
                'tcp_syn_count', 'tcp_psh_count', 'tcp_rst_count', 'iat_avg_packets']

X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
test[numeric_cols] = test[numeric_cols].fillna(X[numeric_cols].mean())

# target 라벨 인코딩
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# LightGBM 데이터셋 객체 생성
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# 파라미터 설정
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}
base_models = {
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=5,
                                             class_weight='balanced', random_state=42, n_jobs=-1),
    "Extra Trees": ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "AdaBoost": AdaBoostClassifier(n_estimators=150, learning_rate=0.1),
    "XGBoost": XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1,
                             subsample=0.8, colsample_bytree=0.8, use_label_encoder=False,
                             eval_metric='mlogloss', random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced')
    "LightGBM":lgb.LGBMClassifier(params, train_data, num_boost_round=100, valid_sets=[test_data])
}
# 각 모델의 파라미터 후보 정의
param_grids = {

    "Random Forest": {
        'n_estimators': [100, 200],
        'max_depth': [10, 15],
        'min_samples_split': [2, 5]
    },
    "Extra Trees": {
        'n_estimators': [100, 200],
        'max_depth': [None, 10]
    },
    "AdaBoost": {
        'n_estimators': [100, 150],
        'learning_rate': [0.05, 0.1]
    },
    "XGBoost": {
        'n_estimators': [200, 300],
        'max_depth': [6, 8],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8],
        'colsample_bytree': [0.8]
    },
    "KNN": {
        'n_neighbors': [5, 7, 9]
    },
    "Logistic Regression": {
        'C': [0.1, 1.0, 10],
        'penalty': ['l2']
    }
    "LightGBM":{
        'params':,
        'train_data':,
        'num_boost_round':
    }
}

# 최적 모델 저장용
best_model = None
best_score = 0
best_model_name = ""

# 그리드 서치 수행
for name, model in base_models.items():
    if name in ["Voting Ensemble", "Stacking Ensemble"]:
        continue  # Voting/Stacking은 후처리로

    print(f"\n GridSearchCV: {name}")

    grid = GridSearchCV(
        model,
        param_grids[name],
        scoring='f1_macro',
        cv=3,
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)
    print("최적 파라미터:", grid.best_params_)

    # 검증 데이터로 평가
    y_val_pred = grid.predict(X_val)
    score = f1_score(y_val, y_val_pred, average='macro')

    if score > best_score:
        best_score = score
        best_model = grid.best_estimator_
        best_model_name = name

# Voting & Stacking은 최적 개별 모델을 기반으로 재정의
voting_ensemble = VotingClassifier(
    estimators=[(name, best_model if name == best_model_name else base_models[name])
                for name in base_models if name not in ["Voting Ensemble", "Stacking Ensemble"]],
    voting='soft',
    n_jobs=-1
)

stacking_ensemble = StackingClassifier(
    estimators=[(name, best_model if name == best_model_name else base_models[name])
                for name in base_models if name not in ["Voting Ensemble", "Stacking Ensemble"]],
    final_estimator=LogisticRegression(max_iter=1000),
    passthrough=True,
    n_jobs=-1
)

# Voting 평가
for ens_name, ens_model in [("Voting Ensemble", voting_ensemble), ("Stacking Ensemble", stacking_ensemble)]:
    print(f"\n▶ {ens_name}")
    ens_model.fit(X_train, y_train)
    y_val_pred = ens_model.predict(X_val)
    score = f1_score(y_val, y_val_pred, average='macro')
    print(f"{ens_name} Macro F1 Score: {score:.4f}")

    if score > best_score:
        best_score = score
        best_model = ens_model
        best_model_name = ens_name

print(f"\n 최종 최적 모델: {best_model_name} (Macro F1: {best_score:.4f})")

# 테스트 예측
test_pred = best_model.predict(test)
test_pred_labels = label_encoder.inverse_transform(test_pred)

sample_submission['attack_type'] = test_pred_labels
sample_submission.to_csv('./best_model_submission.csv', index=False)
