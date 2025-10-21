import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
from sklearn.impute import SimpleImputer

class MLPClassificationModel:
    def __init__(self, train_file_path: str, test_file_path: str):
        """初始化MLP分类模型"""
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path
        self.train_data = None
        self.test_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.transformer = None
        self.model = None
        self.accuracy = None
        self.label_encoder = {}
        
    def load_data(self):
        """加载训练集和测试集数据"""
        try:
            self.train_data = pd.read_csv(self.train_file_path)
            self.test_data = pd.read_csv(self.test_file_path)
            print(f"成功加载训练集数据，共{self.train_data.shape[0]}行{self.train_data.shape[1]}列")
            print(f"成功加载测试集数据，共{self.test_data.shape[0]}行{self.test_data.shape[1]}列")
        except Exception as e:
            print(f"加载数据时出错: {e}")
            raise
    
    def prepare_data(self):
        """准备训练数据和测试数据，处理分类特征"""
        if self.train_data is None or self.test_data is None:
            raise ValueError("请先加载数据")
        
        # 定义特征和目标变量
        X_train = self.train_data.iloc[:, :-2]  # 除最后两列外的所有列作为特征
        y_train_attack = self.train_data['attack_cat']  # 攻击类别
        y_train_label = self.train_data['label']  # 标签
        
        X_test = self.test_data.iloc[:, :-2]  # 除最后两列外的所有列作为特征
        y_test_attack = self.test_data['attack_cat']  # 攻击类别
        y_test_label = self.test_data['label']  # 标签
        
        # 由于需要预测两个目标变量，我们选择预测'label'列
        # 如果需要同时预测两个列，可以修改这里的逻辑
        self.y_train = y_train_label
        self.y_test = y_test_label
        
        # 获取分类特征列
        categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = X_train.select_dtypes(exclude=['object']).columns.tolist()
        
        print(f"分类特征列: {categorical_cols}")
        print(f"数值特征列: {numerical_cols}")
        
        # 创建预处理转换器
        self.transformer = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', StandardScaler())
                ]), numerical_cols),
                ('cat', Pipeline([
                    ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ]), categorical_cols)
            ]
        )
        
        # 处理训练数据
        print("开始处理训练数据...")
        self.X_train = self.transformer.fit_transform(X_train)
        print(f"训练集特征处理完成，形状: {self.X_train.shape}")
        
        # 处理测试数据
        print("开始处理测试数据...")
        self.X_test = self.transformer.transform(X_test)
        print(f"测试集特征处理完成，形状: {self.X_test.shape}")
        
    def build_model(self, hidden_layer_sizes=(100, 50), activation='relu', 
                    solver='adam', max_iter=500, random_state=42):
        """构建MLP分类模型"""
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1,
            verbose=True
        )
        print(f"构建MLP分类模型，隐藏层结构: {hidden_layer_sizes}")
    
    def train_model(self):
        """训练模型"""
        if self.model is None:
            raise ValueError("请先构建模型")
        if self.X_train is None or self.y_train is None:
            raise ValueError("请先准备数据")
        
        print("开始训练模型...")
        start_time = time.time()
        self.model.fit(self.X_train, self.y_train)
        end_time = time.time()
        
        print(f"模型训练完成，耗时: {end_time - start_time:.2f}秒")
        print(f"迭代次数: {self.model.n_iter_}")
        if hasattr(self.model, 'best_loss_') and self.model.best_loss_ is not None:
            print(f"最佳损失值: {self.model.best_loss_:.4f}")
    
    def evaluate_model(self):
        """评估模型"""
        if self.model is None or not hasattr(self.model, 'n_iter_'):
            raise ValueError("请先训练模型")
        if self.X_test is None or self.y_test is None:
            raise ValueError("请先准备数据")
        
        # 在测试集上进行预测
        y_pred = self.model.predict(self.X_test)
        
        # 计算评估指标
        self.accuracy = accuracy_score(self.y_test, y_pred)
        
        print(f"模型评估结果:")
        print(f"准确率: {self.accuracy:.4f}")
        print("\n分类报告:")
        print(classification_report(self.y_test, y_pred))
        
        # 生成混淆矩阵
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title('混淆矩阵')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()
        
        print("混淆矩阵已保存为 'confusion_matrix.png'")
        
        return self.accuracy

from sklearn.pipeline import Pipeline

if __name__ == "__main__":
    # 数据集路径
    train_file_path = 'd:\\code\\大作业\\数据\\UNSW_NB15_training-set.csv'
    test_file_path = 'd:\\code\\大作业\\数据\\UNSW_NB15_testing-set.csv'
    
    # 创建MLP分类模型实例
    mlp_model = MLPClassificationModel(train_file_path, test_file_path)
    
    try:
        # 执行完整的模型流程
        print("===== 开始MLP分类模型训练和评估 =====")
        mlp_model.load_data()
        mlp_model.prepare_data()
        mlp_model.build_model(hidden_layer_sizes=(128, 64, 32))
        mlp_model.train_model()
        accuracy = mlp_model.evaluate_model()
        print(f"\n===== MLP分类模型任务完成 =====")
        print(f"最终准确率: {accuracy:.4f}")
    except Exception as e:
        print(f"发生错误: {e}")