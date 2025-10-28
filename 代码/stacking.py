import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
import matplotlib.pyplot as plt
import seaborn as sns

# 导入基学习器模块
from xgboost_model import create_xgboost_model_for_attack_cat, create_xgboost_model_for_label
from random_forest_model import create_random_forest_model_for_attack_cat, create_random_forest_model_for_label
from catboost_model import create_catboost_model_for_attack_cat, create_catboost_model_for_label

# 设置随机种子以确保结果可复现
np.random.seed(42)

# 加载与读取书记
train_file_path = r'd:\code\大作业\数据\UNSW_NB15_training-set.csv'
test_file_path = r'd:\code\大作业\数据\UNSW_NB15_testing-set.csv'
df_train = pd.read_csv(train_file_path)
df_test = pd.read_csv(test_file_path)

# 数据预处理类
class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder_attack_cat = LabelEncoder()
        self.label_encoder_label = LabelEncoder()
        self.categorical_columns = []
        self.drop_columns = ['id']  # 需要删除的列
        self.label_encoders = {}  # 存储每个分类列的编码器
        self.feature_columns = []  # 存储特征列名，用于特征重要性展示
    
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
        
        # 编码标签并记录类别分布
        if 'attack_cat' in df.columns:
            # 记录attack_cat的类别分布
            attack_cat_counts = df['attack_cat'].value_counts()
            print("attack_cat类别分布:")
            for cat, count in attack_cat_counts.items():
                print(f"  {cat}: {count}个样本 ({count/len(df)*100:.2f}%)")
                           
            self.label_encoder_attack_cat.fit(df['attack_cat'].unique())
            print(f"attack_cat类别总数: {len(self.label_encoder_attack_cat.classes_)}")
            print(f"attack_cat类别列表: {list(self.label_encoder_attack_cat.classes_)}")
            
        if 'label' in df.columns:
            # 将label转换为字符串，确保后续处理正确
            df['label'] = df['label'].astype(str)
            self.label_encoder_label.fit(df['label'].unique())
            print(f"label类别总数: {len(self.label_encoder_label.classes_)}")
            print(f"label类别列表: {list(self.label_encoder_label.classes_)}")
        
        # 准备用于标准化的特征
        self.feature_columns = [col for col in df.columns if col not in self.drop_columns + ['attack_cat', 'label']]
        
        # 拟合标准化器
        self.scaler.fit(df[self.feature_columns])
        
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
        
        # 标准化特征
        X = self.scaler.transform(df[self.feature_columns])
        
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

# Stacking集成学习算法的实现
class StackingClassifier:
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models  # 基学习器列表
        self.meta_model = meta_model    # 元学习器
        self.n_folds = n_folds          # 交叉验证的折数
        self.kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        self.fitted_base_models = []
        self.num_classes = None  # 将在fit时确定
        self.label_encoder = None  # 保存标签编码器用于调试
         
    def fit(self, X, y, label_encoder=None):
        # 确保y是NumPy数组
        if hasattr(y, 'values'):
            y_values = y.values
        else:
            y_values = y
        
        # 保存标签编码器
        self.label_encoder = label_encoder
        
        # 确定类别数量
        self.num_classes = len(np.unique(y_values))
        
        # 打印训练数据中各类别的分布
        #unique_classes, counts = np.unique(y_values, return_counts=True)
        #print(f"训练数据类别分布: {dict(zip(unique_classes, counts))}")
        # 如果有标签编码器，打印类别名称
        if label_encoder and hasattr(label_encoder, 'classes_'):
            print("类别名称映射:")
            for i, cls in enumerate(label_encoder.classes_):
                print(f"  {i}: {cls}")
        
        # 为元学习器创建训练数据 - 对于多分类问题，使用每个基学习器的类别预测和概率
        # 直接使用类别数量来确定概率维度
        proba_dim = self.num_classes
        
        # 元特征将包含每个基学习器的预测类别和所有类别的概率
        meta_features = np.zeros((X.shape[0], len(self.base_models) * (1 + proba_dim)))
        
        # 使用交叉验证训练每个基学习器并生成元特征
        for i, model in enumerate(self.base_models):
            for train_idx, val_idx in self.kf.split(X):
                # 在训练折上训练基学习器
                model.fit(X[train_idx], y_values[train_idx])
                
                # 获取验证折的预测
                y_pred = model.predict(X[val_idx])
                # 确保y_pred是一维的
                y_pred = np.ravel(y_pred)
                
                if hasattr(model, 'predict_proba'):
                    # 获取概率预测
                    proba = model.predict_proba(X[val_idx])
                    
                    # 存储预测类别
                    meta_features[val_idx, i * (1 + proba_dim)] = y_pred
                    
                    # 存储每个类别的概率
                    for j in range(proba_dim):
                        meta_features[val_idx, i * (1 + proba_dim) + 1 + j] = proba[:, j]
                else:
                    # 只有类别预测
                    meta_features[val_idx, i] = y_pred
        
        # 在完整数据集上重新训练所有基学习器
        self.fitted_base_models = []
        for model in self.base_models:
            model_clone = self._clone_model(model)
            model_clone.fit(X, y_values)
            self.fitted_base_models.append(model_clone)
        
        # 训练元学习器
        self.meta_model.fit(meta_features, y_values)
        
        return self
    
    def predict(self, X):
        # 使用已存储的类别数量来确定概率维度
        proba_dim = self.num_classes
        
        # 元特征将包含每个基学习器的预测类别和所有类别的概率
        meta_features = np.zeros((X.shape[0], len(self.fitted_base_models) * (1 + proba_dim)))
        
        # 使用每个基学习器生成预测
        for i, model in enumerate(self.fitted_base_models):
            # 获取预测
            y_pred = model.predict(X)
            # 确保y_pred是一维的
            y_pred = np.ravel(y_pred)
            
            # 获取概率预测
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                
                # 确保概率维度匹配
                if proba.shape[1] != proba_dim:
                    print(f"警告: 模型预测概率维度({proba.shape[1]})与预期维度({proba_dim})不匹配")
                    # 尝试填充缺失的概率（如果预测的类别数少于预期）
                    if proba.shape[1] < proba_dim:
                        padded_proba = np.zeros((proba.shape[0], proba_dim))
                        padded_proba[:, :proba.shape[1]] = proba
                        proba = padded_proba
                    else:
                        # 如果预测的类别数多于预期，只使用前proba_dim个
                        proba = proba[:, :proba_dim]
            else:
                # 如果没有概率输出，创建一个one-hot编码的概率数组
                proba = np.zeros((len(X), proba_dim))
                for j in range(len(X)):
                    if y_pred[j] < proba_dim:  # 确保索引有效
                        proba[j, y_pred[j]] = 1.0
            
            # 存储预测类别
            meta_features[:, i * (1 + proba_dim)] = y_pred
            
            # 存储每个类别的概率
            for j in range(proba_dim):
                meta_features[:, i * (1 + proba_dim) + 1 + j] = proba[:, j]
        
        # 使用元学习器进行最终预测
        return self.meta_model.predict(meta_features)
    
    def _clone_model(self, model):
        # 简单克隆模型的方法
        import copy
        return copy.deepcopy(model)

# 定义基学习器和元学习器
# 为attack_cat预测定义模型
def create_models_for_attack_cat():
    # 1. XGBoost分类器 - 添加scale_pos_weight参数以处理类别不平衡
    xgb_model = create_xgboost_model_for_attack_cat()
    
    # 2. 随机森林分类器 - 添加class_weight='balanced'以处理类别不平衡
    rf_model = create_random_forest_model_for_attack_cat()
    rf_model.set_params(class_weight='balanced')
    
    # 3. CatBoost分类器
    catboost_model = create_catboost_model_for_attack_cat()
    
    # 定义元学习器 - 逻辑回归，添加class_weight='balanced'以处理类别不平衡
    meta_model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    
    # 创建Stacking集成学习模型
    stacking_model = StackingClassifier(
        base_models=[xgb_model, rf_model, catboost_model],
        meta_model=meta_model,
        n_folds=5  # 使用5折交叉验证
    )
    
    models = {
        'XGBoost': xgb_model,
        'Random Forest': rf_model,
        'CatBoost': catboost_model,
        'Stacking': stacking_model
    }
    
    return models

# 为label预测定义模型
def create_models_for_label():
    # 1. XGBoost分类器
    xgb_model = create_xgboost_model_for_label()
    
    # 2. 随机森林分类器
    rf_model = create_random_forest_model_for_label()
    
    # 3. CatBoost分类器
    catboost_model = create_catboost_model_for_label()
    
    # 定义元学习器 - 逻辑回归
    meta_model = LogisticRegression(random_state=42, max_iter=1000)
    
    # 创建Stacking集成学习模型
    stacking_model = StackingClassifier(
        base_models=[xgb_model, rf_model, catboost_model],
        meta_model=meta_model,
        n_folds=5  # 使用5折交叉验证
    )
    
    models = {
        'XGBoost': xgb_model,
        'Random Forest': rf_model,
        'CatBoost': catboost_model,
        'Stacking': stacking_model
    }
    
    return models

# 获取特征重要性
def get_feature_importance(model, model_name, feature_names):
    """
    从模型中提取特征重要性
    """
    importances = None
    
    try:
        # 尝试获取不同模型的特征重要性
        if hasattr(model, 'feature_importances_'):
            # XGBoost, Random Forest, CatBoost
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # Logistic Regression
            # 对于逻辑回归，取系数的绝对值作为重要性度量
            importances = np.abs(model.coef_[0])
        else:
            print(f"警告: 模型 {model_name} 不支持特征重要性提取")
            return None
        
        # 创建特征重要性字典
        importance_dict = dict(zip(feature_names, importances))
        
        # 按重要性排序
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_importance
    except Exception as e:
        print(f"提取 {model_name} 的特征重要性时出错: {e}")
        return None

# 可视化特征重要性
def plot_feature_importance(importance_list, model_name, n_features=10):
    """
    可视化特征重要性
    """
    if not importance_list:
        return
    
    # 只取前n个最重要的特征
    top_features = importance_list[:n_features]
    features = [item[0] for item in top_features]
    importances = [item[1] for item in top_features]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances, y=features)
    plt.title(f'{model_name} - 特征重要性（前{n_features}个）')
    plt.xlabel('重要性分数')
    plt.ylabel('特征')
    plt.tight_layout()
    plt.savefig(f'{model_name}_feature_importance.png')
    plt.close()

# 评估模型
def evaluate_model(model_name, y_true, y_pred, label_encoder):
    # 计算准确率
    acc = accuracy_score(y_true, y_pred)
    
    # 检查所有预测类别
    predicted_classes = np.unique(y_pred)
    true_classes = np.unique(y_true)
    
    # 检查是否有类别在预测中缺失
    missing_classes = [cls for cls in true_classes if cls not in predicted_classes]
    if missing_classes:
        print(f"警告 - {model_name}: 以下类别在预测中完全缺失: {missing_classes}")
        # 将缺失类别的标签名称转换为实际类别名
        if hasattr(label_encoder, 'classes_'):
            missing_class_names = [label_encoder.classes_[cls] if cls < len(label_encoder.classes_) else 'Unknown' 
                                  for cls in missing_classes]
            print(f"缺失类别的名称: {missing_class_names}")
    
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

def main():
    # 初始化数据预处理器
    preprocessor = DataPreprocessor()
    
    # 在训练集上拟合预处理器
    print("拟合数据预处理器...")
    preprocessor.fit(df_train)
    
    # 转换训练集和测试集
    print("转换训练集和测试集...")
    X_train, y_train_attack_cat, y_train_label = preprocessor.transform(df_train)
    X_test, y_test_attack_cat, y_test_label = preprocessor.transform(df_test)
    
    # 打印转换后的数据信息
    print(f"转换后特征维度: {X_train.shape[1]}")
    if y_train_attack_cat is not None:
        print(f"训练集中attack_cat唯一值数量: {len(np.unique(y_train_attack_cat))}")
        print(f"测试集中attack_cat唯一值数量: {len(np.unique(y_test_attack_cat))}")
        # 检查是否有类别在测试集中存在但在训练集中不存在
        train_classes = set(np.unique(y_train_attack_cat))
        test_classes = set(np.unique(y_test_attack_cat))
        new_classes = test_classes - train_classes
        if new_classes:
            print(f"警告: 测试集中存在训练集未见过的类别: {new_classes}")
    if y_train_label is not None:
        print(f"训练集中label唯一值数量: {len(np.unique(y_train_label))}")
        print(f"测试集中label唯一值数量: {len(np.unique(y_test_label))}")
     
    # 创建模型
    attack_cat_models = create_models_for_attack_cat()
    
    # 训练和评估模型
    attack_cat_results = {}
    attack_cat_feature_importances = {}
    
    print("开始训练attack_cat预测模型...")
    for name, model in attack_cat_models.items():
        start_time = time.time()
        
        # 训练模型
        model.fit(X_train, y_train_attack_cat)
        
        # 在测试集上预测
        y_pred = model.predict(X_test)
        
        # 评估模型
        acc, report, cm = evaluate_model(name, y_test_attack_cat, y_pred, preprocessor.label_encoder_attack_cat)
        
        # 获取和存储特征重要性
        if name != 'Stacking':  # 对于Stacking模型，我们分别获取基学习器的特征重要性
            importance = get_feature_importance(model, name, preprocessor.feature_columns)
            if importance:
                attack_cat_feature_importances[name] = importance
                plot_feature_importance(importance, f'{name}_attack_cat')
                print(f"\n{name} 特征重要性（attack_cat预测）:")
                for i, (feature, score) in enumerate(importance[:10]):  # 打印前10个特征
                    print(f"{i+1}. {feature}: {score:.4f}")
        elif name == 'Stacking' and hasattr(model, 'fitted_base_models'):
            # 对于Stacking模型，获取每个基学习器的特征重要性
            for i, base_model in enumerate(model.fitted_base_models):
                base_model_name = list(attack_cat_models.keys())[i]  # 获取基学习器名称
                if base_model_name != 'Stacking':
                    importance = get_feature_importance(base_model, f'{base_model_name}_in_Stacking', preprocessor.feature_columns)
                    if importance:
                        attack_cat_feature_importances[f'{base_model_name}_in_Stacking'] = importance
        
        end_time = time.time()
        
        # 存储结果
        attack_cat_results[name] = {
            'Accuracy': acc,
            'Report': report,
            'Confusion Matrix': cm,
            'Time (s)': end_time - start_time,
            'Feature Importance': attack_cat_feature_importances.get(name) if name != 'Stacking' else None
        }
        
        print(f"{name} - 准确率: {acc:.4f}, 耗时: {end_time - start_time:.2f}秒")
    
    # 创建模型
    label_models = create_models_for_label()
    
    # 训练和评估模型
    label_results = {}
    label_feature_importances = {}
    
    print("开始训练label预测模型...")
    for name, model in label_models.items():
        start_time = time.time()
        
        # 训练模型
        model.fit(X_train, y_train_label)
        
        # 在测试集上预测
        y_pred = model.predict(X_test)
        
        # 评估模型
        acc, report, cm = evaluate_model(name, y_test_label, y_pred, preprocessor.label_encoder_label)
        
        # 获取和存储特征重要性
        if name != 'Stacking':  # 对于Stacking模型，我们分别获取基学习器的特征重要性
            importance = get_feature_importance(model, name, preprocessor.feature_columns)
            if importance:
                label_feature_importances[name] = importance
                plot_feature_importance(importance, f'{name}_label')
                print(f"\n{name} 特征重要性（label预测）:")
                for i, (feature, score) in enumerate(importance[:10]):  # 打印前10个特征
                    print(f"{i+1}. {feature}: {score:.4f}")
        elif name == 'Stacking' and hasattr(model, 'fitted_base_models'):
            # 对于Stacking模型，获取每个基学习器的特征重要性
            for i, base_model in enumerate(model.fitted_base_models):
                base_model_name = list(label_models.keys())[i]  # 获取基学习器名称
                if base_model_name != 'Stacking':
                    importance = get_feature_importance(base_model, f'{base_model_name}_in_Stacking', preprocessor.feature_columns)
                    if importance:
                        label_feature_importances[f'{base_model_name}_in_Stacking'] = importance
        
        end_time = time.time()
        
        # 存储结果
        label_results[name] = {
            'Accuracy': acc,
            'Report': report,
            'Confusion Matrix': cm,
            'Time (s)': end_time - start_time,
            'Feature Importance': label_feature_importances.get(name) if name != 'Stacking' else None
        }
        
        print(f"{name} - 准确率: {acc:.4f}, 耗时: {end_time - start_time:.2f}秒")
    
    # attack_cat预测结果
    print("\nattack_cat预测结果:")
    for name, metrics in attack_cat_results.items():
        print(f"\n{name}:")
        print(f"准确率: {metrics['Accuracy']:.4f}")
        print("分类报告:")
        print(metrics['Report'])
    
    # label预测结果
    print("\nlabel预测结果:")
    for name, metrics in label_results.items():
        print(f"\n{name}:")
        print(f"准确率: {metrics['Accuracy']:.4f}")
        print("分类报告:")
        print(metrics['Report'])
    
    # 输出特征重要性汇总
    print("\n===== 特征重要性汇总 =====")
    
    print("\nattack_cat预测任务特征重要性:")
    for model_name, importance in attack_cat_feature_importances.items():
        if model_name.endswith('_in_Stacking'):
            continue  # 跳过Stacking中的基学习器，避免重复输出
        print(f"\n{model_name} 前5个重要特征:")
        for i, (feature, score) in enumerate(importance[:5]):
            print(f"  {i+1}. {feature}: {score:.4f}")
    
    print("\nlabel预测任务特征重要性:")
    for model_name, importance in label_feature_importances.items():
        if model_name.endswith('_in_Stacking'):
            continue  # 跳过Stacking中的基学习器，避免重复输出
        print(f"\n{model_name} 前5个重要特征:")
        for i, (feature, score) in enumerate(importance[:5]):
            print(f"  {i+1}. {feature}: {score:.4f}")
    
    print("\n模型训练和评估完成! 特征重要性可视化图表已保存为PNG文件。")

# 运行主函数
if __name__ == "__main__":
    main()