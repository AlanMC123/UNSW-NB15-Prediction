import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time

# 导入要测试的模型
from random_forest_model import create_random_forest_model_for_attack_cat, create_random_forest_model_for_label

# 设置随机种子以确保结果可复现
np.random.seed(42)

# 数据预处理类
class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder_attack_cat = LabelEncoder()
        self.label_encoder_label = LabelEncoder()
        self.categorical_columns = []
        self.drop_columns = ['id']  # 需要删除的列
        self.label_encoders = {}  # 存储每个分类列的编码器
    
    def fit(self, df):
        # 处理缺失值
        df = df.fillna(0)
        
        # 识别并处理分类特征
        self.categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        
        # 为每个分类特征创建并拟合LabelEncoder
        for col in self.categorical_columns:
            if col != 'attack_cat':  # attack_cat将在后面单独处理
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le  # 存储已拟合的编码器
        
        # 编码标签
        if 'attack_cat' in df.columns:
            self.label_encoder_attack_cat.fit(df['attack_cat'].unique())
        if 'label' in df.columns:
            # 将label转换为字符串，确保后续处理正确
            df['label'] = df['label'].astype(str)
            self.label_encoder_label.fit(df['label'].unique())
        
        # 准备用于标准化的特征
        feature_columns = [col for col in df.columns if col not in self.drop_columns + ['attack_cat', 'label']]
        
        # 拟合标准化器
        self.scaler.fit(df[feature_columns])
        
        return self
    
    def transform(self, df):
        # 处理缺失值
        df = df.fillna(0)
        
        # 使用已拟合的编码器转换分类特征
        for col in self.categorical_columns:
            if col != 'attack_cat' and col in self.label_encoders:
                le = self.label_encoders[col]
                # 处理测试集中可能出现的新类别
                df[col] = df[col].apply(lambda x: len(le.classes_) if x not in le.classes_ else le.transform([x])[0])
        
        # 准备特征
        feature_columns = [col for col in df.columns if col not in self.drop_columns + ['attack_cat', 'label']]
        
        # 标准化特征
        X = self.scaler.transform(df[feature_columns])
        
        # 处理目标变量
        y_attack_cat = None
        y_label = None
        
        if 'attack_cat' in df.columns:
            # 处理可能的新类别
            df['attack_cat'] = df['attack_cat'].apply(
                lambda x: x if x in self.label_encoder_attack_cat.classes_ else 'Unknown'
            )
            y_attack_cat = self.label_encoder_attack_cat.transform(df['attack_cat'])
        
        if 'label' in df.columns:
            # 将label转换为字符串，确保与训练时一致
            df['label'] = df['label'].astype(str)
            # 处理可能的新类别
            df['label'] = df['label'].apply(
                lambda x: x if x in self.label_encoder_label.classes_ else 'Unknown'
            )
            y_label = self.label_encoder_label.transform(df['label'])
        
        return X, y_attack_cat, y_label

# 评估模型
def evaluate_model(model_name, y_true, y_pred, label_encoder):
    # 计算准确率
    acc = accuracy_score(y_true, y_pred)
    
    # 确保target_names是字符串类型
    try:
        # 尝试使用label_encoder的classes_
        target_names = label_encoder.classes_
        # 转换为字符串类型
        target_names = [str(name) for name in target_names]
        report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
    except Exception as e:
        # 如果出现问题，不使用target_names
        print(f"创建分类报告时出错: {e}")
        report = classification_report(y_true, y_pred, zero_division=0)
    
    # 生成混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    return acc, report, cm

# 主测试函数
def main():
    print("开始测试随机森林模型...")
    
    # 加载数据集
    train_file_path = r'd:\code\大作业\数据\UNSW_NB15_training-set.csv'
    test_file_path = r'd:\code\大作业\数据\UNSW_NB15_testing-set.csv'
    
    print(f"加载训练集: {train_file_path}")
    print(f"加载测试集: {test_file_path}")
    
    # 读取训练集和测试集
    df_train = pd.read_csv(train_file_path)
    df_test = pd.read_csv(test_file_path)
    
    print(f"训练集形状: {df_train.shape}")
    print(f"测试集形状: {df_test.shape}")
    
    # 初始化数据预处理器
    preprocessor = DataPreprocessor()
    
    # 在训练集上拟合预处理器
    print("拟合数据预处理器...")
    preprocessor.fit(df_train)
    
    # 转换训练集和测试集
    print("转换训练集和测试集...")
    X_train, y_train_attack_cat, y_train_label = preprocessor.transform(df_train)
    X_test, y_test_attack_cat, y_test_label = preprocessor.transform(df_test)
    
    print(f"特征维度: {X_train.shape[1]}")
    
    # 1. 测试attack_cat预测模型
    print("\n===== 测试attack_cat预测模型 =====")
    
    # 创建模型
    attack_cat_model = create_random_forest_model_for_attack_cat(verbose=2)
    
    # 输出模型参数
    print("模型参数:")
    for param, value in attack_cat_model.get_params().items():
        print(f"  {param}: {value}")
    
    # 训练模型
    print("训练attack_cat预测模型...")
    start_time = time.time()
    # 启用详细输出
    attack_cat_model.fit(X_train, y_train_attack_cat)
    training_time = time.time() - start_time
    print(f"训练完成，耗时: {training_time:.2f}秒")
    
    # 输出模型详细信息
    print("\n模型详细信息:")
    print(f"树的数量: {attack_cat_model.n_estimators}")
    print(f"最大深度: {attack_cat_model.max_depth}")
    print(f"最小样本分割数: {attack_cat_model.min_samples_split}")
    print(f"最小叶节点样本数: {attack_cat_model.min_samples_leaf}")
    print(f"最大特征数: {attack_cat_model.max_features}")
    
    # 输出特征重要性
    print("\n特征重要性（前15个）:")
    if hasattr(attack_cat_model, 'feature_importances_'):
        importances = attack_cat_model.feature_importances_
        # 获取特征名称
        feature_columns = [col for col in df_train.columns if col not in preprocessor.drop_columns + ['attack_cat', 'label']]
        # 按重要性排序
        indices = np.argsort(importances)[::-1][:15]  # 前15个最重要的特征
        for f in range(len(indices)):
            idx = indices[f]
            print(f"  {f + 1}. {feature_columns[idx]}: {importances[idx]:.6f}")
    
    # 在测试集上预测
    print("在测试集上进行预测...")
    y_pred_attack_cat = attack_cat_model.predict(X_test)
    
    # 评估模型
    acc_attack_cat, report_attack_cat, cm_attack_cat = evaluate_model(
        "Random Forest (attack_cat)", 
        y_test_attack_cat, 
        y_pred_attack_cat, 
        preprocessor.label_encoder_attack_cat
    )
    
    print(f"准确率: {acc_attack_cat:.4f}")
    print("分类报告:")
    print(report_attack_cat)
    
    # 2. 测试label预测模型
    print("\n===== 测试label预测模型 =====")
    
    # 创建模型
    label_model = create_random_forest_model_for_label(verbose=2)
    
    # 输出模型参数
    print("模型参数:")
    for param, value in label_model.get_params().items():
        print(f"  {param}: {value}")
    
    # 训练模型
    print("训练label预测模型...")
    start_time = time.time()
    # 启用详细输出
    label_model.fit(X_train, y_train_label)
    training_time = time.time() - start_time
    print(f"训练完成，耗时: {training_time:.2f}秒")
    
    # 输出模型详细信息
    print("\n模型详细信息:")
    print(f"树的数量: {label_model.n_estimators}")
    print(f"最大深度: {label_model.max_depth}")
    print(f"最小样本分割数: {label_model.min_samples_split}")
    print(f"最小叶节点样本数: {label_model.min_samples_leaf}")
    print(f"最大特征数: {label_model.max_features}")
    
    # 输出特征重要性
    print("\n特征重要性（前15个）:")
    if hasattr(label_model, 'feature_importances_'):
        importances = label_model.feature_importances_
        # 获取特征名称
        feature_columns = [col for col in df_train.columns if col not in preprocessor.drop_columns + ['attack_cat', 'label']]
        # 按重要性排序
        indices = np.argsort(importances)[::-1][:15]  # 前15个最重要的特征
        for f in range(len(indices)):
            idx = indices[f]
            print(f"  {f + 1}. {feature_columns[idx]}: {importances[idx]:.6f}")
    
    # 在测试集上预测
    print("在测试集上进行预测...")
    y_pred_label = label_model.predict(X_test)
    
    # 评估模型
    acc_label, report_label, cm_label = evaluate_model(
        "Random Forest (label)", 
        y_test_label, 
        y_pred_label, 
        preprocessor.label_encoder_label
    )
    
    print(f"准确率: {acc_label:.4f}")
    print("分类报告:")
    print(report_label)
    
    print("\n测试完成!")

# 运行测试
if __name__ == "__main__":
    main()