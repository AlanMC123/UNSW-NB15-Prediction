import xgboost as xgb

# XGBoost分类器 - 用于attack_cat预测
def create_xgboost_model_for_attack_cat():
    # 移除硬编码的num_class参数，让XGBoost在fit时自动确定
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='multi:softmax'
    )
    return xgb_model

# XGBoost分类器 - 用于label预测
def create_xgboost_model_for_label():
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='binary:logistic'
    )
    return xgb_model