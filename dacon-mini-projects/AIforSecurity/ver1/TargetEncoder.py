#범주형 인코딩 방식 적용 
#LabelEcorder 대신 TargetEncoder 사용 - 데이터 전처리 최적화
from category_encoders import TargetEncoder

for col in category_features:
    encoder = TargetEncoder()
    X_train[col] = encoder.fit_transform(X_train[col], y_train)
    X_val[col] = encoder.transform(X_val[col])
    test[col] = encoder.transform(test[col])
    ###
