import xgboost as xgb

# XGBoost分类器 - 用于attack_cat预测
def create_xgboost_model_for_attack_cat(verbose=True):
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='multi:softmax',
        verbosity=1 if verbose else 0
    )
    return xgb_model

# XGBoost分类器 - 用于label预测
def create_xgboost_model_for_label(verbose=True):
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='binary:logistic',
        verbosity=1 if verbose else 0
    )
    return xgb_model