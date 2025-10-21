from catboost import CatBoostClassifier

# CatBoost分类器 - 用于attack_cat预测
def create_catboost_model_for_attack_cat():
    catboost_model = CatBoostClassifier(
        iterations=100,
        depth=6,
        learning_rate=0.1,
        loss_function='MultiClass',
        verbose=False,
        random_state=42
    )
    return catboost_model

# CatBoost分类器 - 用于label预测
def create_catboost_model_for_label():
    catboost_model = CatBoostClassifier(
        iterations=100,
        depth=6,
        learning_rate=0.1,
        loss_function='Logloss',
        verbose=False,
        random_state=42
    )
    return catboost_model