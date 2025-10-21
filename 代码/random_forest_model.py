from sklearn.ensemble import RandomForestClassifier

# 随机森林分类器 - 用于attack_cat预测
def create_random_forest_model_for_attack_cat():
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42
    )
    return rf_model

# 随机森林分类器 - 用于label预测
def create_random_forest_model_for_label():
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42
    )
    return rf_model