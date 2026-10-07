#앙상블 모델 적용 
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score

# ...

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42
)

ensemble_model = VotingClassifier(
    estimators=[
        ('rf', rf_model),
        ('xgb', xgb_model)
    ],
    voting='soft'
)

models = {
    "Random Forest": rf_model,
    "XGBoost": xgb_model,
    "Voting Ensemble": ensemble_model
}

best_model = None
best_f1 = 0

for name, model in models.items():
    print(f"\n=== Training {name} ===")
    
    model.fit(X_train, y_train)
    
    y_val_pred = model.predict(X_val)
    
    print(classification_report(y_val, y_val_pred))
    macro_f1 = f1_score(y_val, y_val_pred, average='macro')
    print(f" {name} Macro F1: {macro_f1:.4f}")
    
    if macro_f1 > best_f1:
        best_f1 = macro_f1
        best_model = model
        best_model_name = name

print(f"\n 최적 모델: {best_model_name} (Macro F1: {best_f1:.4f})")

test_pred = best_model.predict(test)
test_pred_labels = label_encoder.inverse_transform(test_pred)

sample_submission['attack_type'] = test_pred_labels
sample_submission.to_csv('./best_model_submission.csv', index=False)
