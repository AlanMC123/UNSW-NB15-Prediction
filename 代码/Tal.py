import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report, confusion_matrix
import time
import os
from pytorch_tabnet.tab_model import TabNetRegressor, TabNetClassifier
import torch

class TabNetRegressionModel:
    def __init__(self, file_path: str):
        """初始化TabNet回归模型"""
        self.file_path = file_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.model = None
        self.r2_score = None
        self.mse = None
        self.feature_names = None
    
    def load_data(self):
        """加载数据集"""
        try:
            self.data = pd.read_csv(self.file_path)
            print(f"成功加载数据，共{self.data.shape[0]}行{self.data.shape[1]}列")
        except Exception as e:
            print(f"加载数据时出错: {e}")
            raise
    
    def prepare_data(self, test_size: float = 0.2, random_state: int = 42):
        """准备训练数据和测试数据"""
        if self.data is None:
            raise ValueError("请先加载数据")
        
        # 定义特征和目标变量
        X = self.data.drop('job_satisfaction_score', axis=1)
        y = self.data['job_satisfaction_score'].values.reshape(-1, 1)
        
        # 保存特征名称
        self.feature_names = X.columns.tolist()
        
        # 检查数据类型
        print(f"特征数据类型: {X.dtypes.value_counts()}")
        
        # 划分训练集和测试集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # 特征标准化 - 确保转换为numpy数组
        self.X_train = self.scaler.fit_transform(self.X_train).astype(np.float32)
        self.X_test = self.scaler.transform(self.X_test).astype(np.float32)
        
        print(f"训练集大小: {self.X_train.shape[0]}，测试集大小: {self.X_test.shape[0]}")
        print(f"训练集特征维度: {self.X_train.shape[1]}")

#接下来是模型的参数部分，这些参数可以根据实际情况进行调整
# n_d: 决策层的特征维度
# n_a: 注意力层的特征维度
# n_steps: 决策步骤数
# gamma: 特征选择的系数
# optimizer_fn: 优化器
# optimizer_params: 优化器参数
# scheduler_fn: 学习率调度器
# scheduler_params: 学习率调度器参数
# mask_type: 掩码类型
# n_shared: 共享Gated Linear Unit的数量
# n_independent: 独立Gated Linear Unit的数量

    def build_model(self, n_d: int = 8, n_a: int = 8, n_steps: int = 3, 
                    gamma: float = 1.3, optimizer_fn: torch.optim.Optimizer = torch.optim.Adam,
                    #optimizer_params: dict = None, 
                    scheduler_fn: torch.optim.lr_scheduler = None,
                    optimizer_params={'lr': 0.001},
                    lambda_sparse=0.001,
                    scheduler_params: dict = None, mask_type: str = 'sparsemax',
                    n_shared: int = 2, n_independent: int = 2):
        """构建TabNet回归模型"""
        if optimizer_params is None:
            optimizer_params = {'lr': 0.01}
        
        # 不使用分类特征参数，因为我们的数据已经过预处理
        self.model = TabNetRegressor(
            n_d=n_d,  # 决策层的特征维度
            n_a=n_a,  # 注意力层的特征维度
            n_steps=n_steps,  # 决策步骤数
            gamma=gamma,  # 特征选择的系数
            cat_idxs=[],  # 没有分类特征（已预处理为数值）
            cat_dims=[],  # 没有分类特征
            cat_emb_dim=1,  # 不影响，因为没有分类特征
            optimizer_fn=optimizer_fn,  # 优化器
            optimizer_params=optimizer_params,  # 优化器参数
            scheduler_fn=scheduler_fn,  # 学习率调度器
            scheduler_params=scheduler_params,  # 学习率调度器参数
            mask_type=mask_type,  # 掩码类型
            n_shared=n_shared,  # 共享Gated Linear Unit的数量
            n_independent=n_independent,  # 独立Gated Linear Unit的数量
            verbose=1  # 详细输出
        )
        
        print(f"构建TabNet回归模型，配置: n_d={n_d}, n_a={n_a}, n_steps={n_steps}, lr={optimizer_params['lr']}")
    
    def train_model(self, max_epochs: int = 50, patience: int = 10, batch_size: int = 512, 
                    virtual_batch_size: int = 64):
        """训练模型"""
        if self.model is None:
            raise ValueError("请先构建模型")
        if self.X_train is None or self.y_train is None:
            raise ValueError("请先准备数据")
        
        print("开始训练TabNet模型...")
        print(f"训练数据形状: X={self.X_train.shape}, y={self.y_train.shape}")
        
        # 确保数据类型正确
        try:
            self.X_train = np.array(self.X_train, dtype=np.float32)
            self.y_train = np.array(self.y_train, dtype=np.float32)
            self.X_test = np.array(self.X_test, dtype=np.float32)
            self.y_test = np.array(self.y_test, dtype=np.float32)
            
            start_time = time.time()
            
            # 训练模型 - 使用TabNet支持的评估指标
            self.model.fit(
                X_train=self.X_train,
                y_train=self.y_train,
                eval_set=[(self.X_train, self.y_train), (self.X_test, self.y_test)],
                eval_name=['train', 'val'],
                eval_metric=['mse', 'rmse'],  # 使用TabNet支持的指标
                max_epochs=max_epochs,
                patience=patience,
                batch_size=batch_size,
                virtual_batch_size=virtual_batch_size,
                num_workers=0,
                drop_last=False
            )
            
            end_time = time.time()
            print(f"模型训练完成，耗时: {end_time - start_time:.2f}秒")
        except Exception as e:
            print(f"训练过程中出错: {e}")
            print(f"数据类型检查: X_train={self.X_train.dtype}, y_train={self.y_train.dtype}")
            import traceback
            traceback.print_exc()
            raise
    
    def evaluate_model(self):
        """评估模型并计算R方值"""
        if self.model is None:
            raise ValueError("请先训练模型")
        if self.X_test is None or self.y_test is None:
            raise ValueError("请先准备数据")
        
        # 在测试集上进行预测
        y_pred = self.model.predict(self.X_test)
        
        # 计算评估指标
        self.r2_score = r2_score(self.y_test, y_pred)
        self.mse = mean_squared_error(self.y_test, y_pred)
        
        print(f"模型评估结果:")
        print(f"R方值: {self.r2_score:.4f}")
        print(f"均方误差(MSE): {self.mse:.4f}")
        
        return self.r2_score
    
    def visualize_results(self, output_dir: str = './visualizations'):
        """可视化模型结果"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 预测值与实际值的散点图
        y_pred = self.model.predict(self.X_test)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_test, y_pred, alpha=0.5)
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        plt.xlabel('实际值 (job_satisfaction_score)')
        plt.ylabel('预测值')
        plt.title(f'预测值 vs 实际值 (R² = {self.r2_score:.4f})')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'tabnet_prediction_vs_actual.png'))
        plt.close()
        
        # 残差图
        residuals = self.y_test - y_pred
        
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('预测值')
        plt.ylabel('残差')
        plt.title('TabNet 残差图')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'tabnet_residuals_plot.png'))
        plt.close()
        
        print(f"可视化结果已保存至: {output_dir}")
    
    def feature_importance(self):
        """分析特征重要性"""
        if self.model is None:
            raise ValueError("请先训练模型")
        
        # 获取特征重要性
        feature_importances = self.model.feature_importances_
        
        # 创建特征重要性DataFrame
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': feature_importances
        })
        
        # 按重要性排序
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # 绘制特征重要性条形图
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(15))
        plt.title('TabNet Top 15 特征重要性')
        plt.tight_layout()
        plt.savefig('tabnet_feature_importance.png')
        plt.close()
        
        print("\nTabNet特征重要性分析:")
        print(importance_df.head(10))
        
        return importance_df
    
    def plot_attentions(self, output_dir: str = './visualizations'):
        """绘制TabNet的注意力可视化"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 获取前10个样本的注意力权重
        if self.X_test.shape[0] > 10:
            sample_indices = np.random.choice(self.X_test.shape[0], 10, replace=False)
        else:
            sample_indices = np.arange(self.X_test.shape[0])
        
        X_sample = self.X_test[sample_indices]
        
        # 获取注意力掩码
        masks = self.model.explain(X_sample)[0]
        
        # 为每个样本绘制注意力掩码
        for i, mask in enumerate(masks):
            plt.figure(figsize=(12, 8))
            plt.imshow(mask.T, aspect='auto', cmap='Blues', origin='lower')
            plt.colorbar(label='Attention')
            plt.xlabel('决策步骤')
            plt.ylabel('特征索引')
            plt.title(f'样本 {sample_indices[i]} 的注意力掩码')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'tabnet_attention_sample_{i}.png'))
            plt.close()
            
            # 如果只绘制一个样本就退出（避免过多图表）
            if i == 0:
                break
        
        print(f"注意力可视化结果已保存至: {output_dir}")

class TabNetClassificationModel:
    def __init__(self, train_file_path: str, test_file_path: str):
        """初始化TabNet分类模型"""
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path
        self.train_data = None
        self.test_data = None
        self.X_train = None
        self.X_test = None
        self.y_train_attack_cat = None
        self.y_test_attack_cat = None
        self.y_train_label = None
        self.y_test_label = None
        self.scaler = StandardScaler()
        self.encoder_attack_cat = OneHotEncoder(handle_unknown='ignore')
        self.model_attack_cat = None
        self.model_label = None
        self.accuracy_attack_cat = None
        self.accuracy_label = None
        self.feature_names = None
        self.cat_columns = None
        self.num_columns = None
        self.encoder = OneHotEncoder(handle_unknown='ignore')
    
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
        """准备训练数据和测试数据"""
        if self.train_data is None or self.test_data is None:
            raise ValueError("请先加载数据")
        
        # 定义特征和目标变量（最后两列是attack_cat和label）
        X_train = self.train_data.iloc[:, :-2]
        X_test = self.test_data.iloc[:, :-2]
        
        # 目标变量
        self.y_train_attack_cat = self.train_data['attack_cat'].values
        self.y_test_attack_cat = self.test_data['attack_cat'].values
        self.y_train_label = self.train_data['label'].values
        self.y_test_label = self.test_data['label'].values
        
        # 保存特征名称
        self.feature_names = X_train.columns.tolist()
        
        # 识别分类特征和数值特征
        self.cat_columns = X_train.select_dtypes(include=['object']).columns.tolist()
        self.num_columns = X_train.select_dtypes(exclude=['object']).columns.tolist()
        
        print(f"分类特征数量: {len(self.cat_columns)}, 数值特征数量: {len(self.num_columns)}")
        print(f"分类特征: {self.cat_columns}")
        
        # 对分类特征进行One-Hot编码
        if len(self.cat_columns) > 0:
            # 处理训练集
            X_train_cat_encoded = self.encoder.fit_transform(X_train[self.cat_columns]).toarray()
            # 处理测试集
            X_test_cat_encoded = self.encoder.transform(X_test[self.cat_columns]).toarray()
            
            # 组合编码后的分类特征和数值特征
            X_train_encoded = np.hstack([X_train[self.num_columns].values, X_train_cat_encoded])
            X_test_encoded = np.hstack([X_test[self.num_columns].values, X_test_cat_encoded])
        else:
            # 如果没有分类特征，直接使用数值特征
            X_train_encoded = X_train[self.num_columns].values
            X_test_encoded = X_test[self.num_columns].values
        
        # 特征标准化
        self.X_train = self.scaler.fit_transform(X_train_encoded).astype(np.float32)
        self.X_test = self.scaler.transform(X_test_encoded).astype(np.float32)
        
        print(f"训练集大小: {self.X_train.shape[0]}，测试集大小: {self.X_test.shape[0]}")
        print(f"训练集特征维度: {self.X_train.shape[1]}")
    
    def build_model(self, n_d: int = 8, n_a: int = 8, n_steps: int = 3, 
                    gamma: float = 1.3, optimizer_fn: torch.optim.Optimizer = torch.optim.Adam,
                    optimizer_params={'lr': 0.001}, 
                    scheduler_fn: torch.optim.lr_scheduler = None,
                    scheduler_params: dict = None, mask_type: str = 'sparsemax',
                    n_shared: int = 2, n_independent: int = 2):
        """构建TabNet分类模型"""
        # 构建用于预测attack_cat的模型
        self.model_attack_cat = TabNetClassifier(
            n_d=n_d,  # 决策层的特征维度
            n_a=n_a,  # 注意力层的特征维度
            n_steps=n_steps,  # 决策步骤数
            gamma=gamma,  # 特征选择的系数
            optimizer_fn=optimizer_fn,  # 优化器
            optimizer_params=optimizer_params,  # 优化器参数
            scheduler_fn=scheduler_fn,  # 学习率调度器
            scheduler_params=scheduler_params,  # 学习率调度器参数
            mask_type=mask_type,  # 掩码类型
            n_shared=n_shared,  # 共享Gated Linear Unit的数量
            n_independent=n_independent,  # 独立Gated Linear Unit的数量
            verbose=1  # 详细输出
        )
        
        # 构建用于预测label的模型
        self.model_label = TabNetClassifier(
            n_d=n_d,  # 决策层的特征维度
            n_a=n_a,  # 注意力层的特征维度
            n_steps=n_steps,  # 决策步骤数
            gamma=gamma,  # 特征选择的系数
            optimizer_fn=optimizer_fn,  # 优化器
            optimizer_params=optimizer_params,  # 优化器参数
            scheduler_fn=scheduler_fn,  # 学习率调度器
            scheduler_params=scheduler_params,  # 学习率调度器参数
            mask_type=mask_type,  # 掩码类型
            n_shared=n_shared,  # 共享Gated Linear Unit的数量
            n_independent=n_independent,  # 独立Gated Linear Unit的数量
            verbose=1  # 详细输出
        )
        
        print(f"构建TabNet分类模型，配置: n_d={n_d}, n_a={n_a}, n_steps={n_steps}, lr={optimizer_params['lr']}")
    
    def train_model(self, max_epochs: int = 50, patience: int = 10, batch_size: int = 512, 
                    virtual_batch_size: int = 64):
        """训练模型"""
        if self.model_attack_cat is None or self.model_label is None:
            raise ValueError("请先构建模型")
        if self.X_train is None or self.y_train_attack_cat is None or self.y_train_label is None:
            raise ValueError("请先准备数据")
        
        # 确保数据类型正确
        try:
            self.X_train = np.array(self.X_train, dtype=np.float32)
            # 确保目标变量是整数类型
            # 处理attack_cat：如果是字符串类型，将其转换为整数编码
            if isinstance(self.y_train_attack_cat[0], str):
                from sklearn.preprocessing import LabelEncoder
                le_attack_cat = LabelEncoder()
                self.y_train_attack_cat = le_attack_cat.fit_transform(self.y_train_attack_cat)
                self.y_test_attack_cat = le_attack_cat.transform(self.y_test_attack_cat)
            
            self.y_train_attack_cat = self.y_train_attack_cat.astype(np.int64)
            self.y_train_label = self.y_train_label.astype(np.int64)
            self.X_test = np.array(self.X_test, dtype=np.float32)
            self.y_test_attack_cat = self.y_test_attack_cat.astype(np.int64)
            self.y_test_label = self.y_test_label.astype(np.int64)
            
            # 检查attack_cat是否有多个类别
            if len(np.unique(self.y_train_attack_cat)) > 1:
                print("开始训练预测attack_cat的TabNet模型...")
                start_time = time.time()
                
                # 训练模型 - 使用TabNet支持的评估指标
                self.model_attack_cat.fit(
                    X_train=self.X_train,
                    y_train=self.y_train_attack_cat,
                    eval_set=[(self.X_train, self.y_train_attack_cat), (self.X_test, self.y_test_attack_cat)],
                    eval_name=['train', 'val'],
                    eval_metric=['accuracy'],  # 使用准确率作为评估指标
                    max_epochs=max_epochs,
                    patience=patience,
                    batch_size=batch_size,
                    virtual_batch_size=virtual_batch_size,
                    num_workers=0,
                    drop_last=False
                )
                
                end_time = time.time()
                print(f"attack_cat模型训练完成，耗时: {end_time - start_time:.2f}秒")
            else:
                print("attack_cat只有一个类别，跳过训练该模型")
            
            # 训练预测label的模型
            print("开始训练预测label的TabNet模型...")
            start_time = time.time()
            
            # 训练模型 - 使用TabNet支持的评估指标
            self.model_label.fit(
                X_train=self.X_train,
                y_train=self.y_train_label,
                eval_set=[(self.X_train, self.y_train_label), (self.X_test, self.y_test_label)],
                eval_name=['train', 'val'],
                eval_metric=['accuracy'],  # 使用准确率作为评估指标
                max_epochs=max_epochs,
                patience=patience,
                batch_size=batch_size,
                virtual_batch_size=virtual_batch_size,
                num_workers=0,
                drop_last=False
            )
            
            end_time = time.time()
            print(f"label模型训练完成，耗时: {end_time - start_time:.2f}秒")
        except Exception as e:
            print(f"训练过程中出错: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def evaluate_model(self):
        """评估模型并计算准确率"""
        if self.model_attack_cat is None and self.model_label is None:
            raise ValueError("请先训练模型")
        if self.X_test is None:
            raise ValueError("请先准备数据")
        
        # 评估attack_cat模型
        if len(np.unique(self.y_train_attack_cat)) > 1:
            # 在测试集上进行预测
            y_pred_attack_cat = self.model_attack_cat.predict(self.X_test)
            
            # 计算评估指标
            self.accuracy_attack_cat = accuracy_score(self.y_test_attack_cat, y_pred_attack_cat)
            
            print(f"attack_cat模型评估结果:")
            print(f"准确率: {self.accuracy_attack_cat:.4f}")
            print("分类报告:")
            print(classification_report(self.y_test_attack_cat, y_pred_attack_cat))
        else:
            print("attack_cat只有一个类别，无法评估该模型")
        
        # 评估label模型
        # 在测试集上进行预测
        y_pred_label = self.model_label.predict(self.X_test)
        
        # 计算评估指标
        self.accuracy_label = accuracy_score(self.y_test_label, y_pred_label)
        
        print(f"\nlabel模型评估结果:")
        print(f"准确率: {self.accuracy_label:.4f}")
        print("分类报告:")
        print(classification_report(self.y_test_label, y_pred_label))
        
        # 绘制混淆矩阵
        self._plot_confusion_matrix(self.y_test_label, y_pred_label, "label")
        
        return self.accuracy_label
    
    def _plot_confusion_matrix(self, y_true, y_pred, target_name):
        """绘制混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('预测标签')
        plt.ylabel('实际标签')
        plt.title(f'{target_name}混淆矩阵')
        plt.tight_layout()
        plt.savefig(f'tabnet_confusion_matrix_{target_name}.png')
        plt.close()
        
        print(f"{target_name}混淆矩阵已保存为'tabnet_confusion_matrix_{target_name}.png'")

if __name__ == "__main__":
    # 数据集路径 - 使用原始字符串避免Unicode转义错误
    train_file_path = r'd:\code\大作业\数据\UNSW_NB15_training-set.csv'
    test_file_path = r'd:\code\大作业\数据\UNSW_NB15_testing-set.csv'
    
    # 创建TabNet分类模型实例
    tabnet_model = TabNetClassificationModel(train_file_path, test_file_path)
    
    try:
        # 执行完整的模型流程
        print("===== 开始TabNet分类模型训练和评估 =====")
        tabnet_model.load_data()
        
        # 查看数据前几行和信息
        print("\n训练集数据前5行:")
        print(tabnet_model.train_data.head())
        print("\n训练集数据信息:")
        print(tabnet_model.train_data.info())
        
        tabnet_model.prepare_data()
        tabnet_model.build_model(
            n_d=8,
            n_a=8,
            n_steps=3,
            gamma=1.3,
            optimizer_params={'lr': 0.01}
        )
        # 使用合适的参数进行训练
        tabnet_model.train_model(max_epochs=50, patience=15, batch_size=1024)
        accuracy = tabnet_model.evaluate_model()
        
        print(f"\n===== TabNet分类模型任务完成 =====")
        print(f"最终准确率: {accuracy:.4f}")
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()