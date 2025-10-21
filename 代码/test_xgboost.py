import pandas as pd
import numpy as np
import time
from sklearn.metrics import classification_report, accuracy_score
from xgboost_model import create_xgboost_model_for_attack_cat, create_xgboost_model_for_label

# 数据预处理类
class DataPreprocessor:
    def __init__(self):
        # 存储标签编码器
        self.label_encoders = {}
        # 需要编码的分类特征列
        self.categorical_columns = ['proto', 'service', 'state', 'attack_cat']
    
    def fit_transform(self, df):
        """训练集预处理"""
        df = df.copy()
        
        # 处理缺失值
        if 'attack_cat' in df.columns:
            df['attack_cat'].fillna('Normal', inplace=True)
        
        # 对分类特征进行标签编码
        for col in self.categorical_columns:
            if col in df.columns:
                self.label_encoders[col] = {}
                unique_values = df[col].unique()
                for i, val in enumerate(unique_values):
                    self.label_encoders[col][val] = i
                df[col] = df[col].map(self.label_encoders[col])
        
        return df
    
    def transform(self, df):
        """测试集预处理"""
        df = df.copy()
        
        # 处理缺失值
        if 'attack_cat' in df.columns:
            df['attack_cat'].fillna('Normal', inplace=True)
        
        # 对分类特征进行标签编码
        for col in self.categorical_columns:
            if col in df.columns:
                # 对于测试集中未见过的值，统一编码为-1
                df[col] = df[col].map(lambda x: self.label_encoders[col].get(x, -1))
        
        return df

# 评估模型性能
def evaluate_model(model, X_test, y_test, label_names=None):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=label_names)
    return accuracy, report

def main():
    # 加载数据集
    train_path = r'd:\code\大作业\数据\UNSW_NB15_training-set.csv'
    test_path = r'd:\code\大作业\数据\UNSW_NB15_testing-set.csv'
    
    print("正在加载数据集...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # 数据预处理
    preprocessor = DataPreprocessor()
    train_df_processed = preprocessor.fit_transform(train_df)
    test_df_processed = preprocessor.transform(test_df)
    
    # 准备特征和标签
    # 排除id、attack_cat和label列作为特征
    feature_columns = [col for col in train_df_processed.columns if col not in ['id', 'attack_cat', 'label']]
    
    X_train = train_df_processed[feature_columns]
    X_test = test_df_processed[feature_columns]
    
    # 测试attack_cat预测模型
    print("\n===== 测试attack_cat预测模型 =====")
    y_train_attack = train_df_processed['attack_cat']
    y_test_attack = test_df_processed['attack_cat']
    
    # 获取attack_cat的标签名称
    if 'attack_cat' in preprocessor.label_encoders:
        attack_cat_names = {v: k for k, v in preprocessor.label_encoders['attack_cat'].items()}
        attack_label_names = [attack_cat_names[i] for i in sorted(attack_cat_names.keys())]
    else:
        attack_label_names = None
    
    # 注意：需要根据实际数据的类别数量调整num_class参数
    # 先创建模型
    attack_model = create_xgboost_model_for_attack_cat()
    
    # 根据实际数据调整num_class参数
    num_classes = len(train_df_processed['attack_cat'].unique())
    if hasattr(attack_model, 'set_params'):
        attack_model.set_params(num_class=num_classes)
    
    print(f"正在训练attack_cat预测模型（{num_classes}个类别）...")
    start_time = time.time()
    attack_model.fit(X_train, y_train_attack)
    train_time = time.time() - start_time
    print(f"模型训练完成，耗时: {train_time:.2f}秒")
    
    # 评估模型
    accuracy, report = evaluate_model(attack_model, X_test, y_test_attack, attack_label_names)
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    
    # 测试label预测模型
    print("\n===== 测试label预测模型 =====")
    y_train_label = train_df_processed['label']
    y_test_label = test_df_processed['label']
    
    # 创建并训练模型
    label_model = create_xgboost_model_for_label()
    print("正在训练label预测模型...")
    start_time = time.time()
    label_model.fit(X_train, y_train_label)
    train_time = time.time() - start_time
    print(f"模型训练完成，耗时: {train_time:.2f}秒")
    
    # 评估模型
    accuracy, report = evaluate_model(label_model, X_test, y_test_label, ['Normal', 'Attack'])
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)

if __name__ == "__main__":
    main()