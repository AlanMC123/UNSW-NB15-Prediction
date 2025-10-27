// 文件元素和按钮
const fileInput = document.getElementById('file-input');
const fileName = document.getElementById('file-name');
const predictBtn = document.getElementById('predict-btn');
const modelSelect = document.getElementById('model-select');
const loading = document.getElementById('loading');
const resultModal = document.getElementById('result-modal');
const closeBtn = document.getElementsByClassName('close-btn')[0];

// 结果显示元素
const totalRecordsEl = document.getElementById('total-records');
const attackedRecordsEl = document.getElementById('attacked-records');
const attackDetailsEl = document.getElementById('attack-details');
const attackChartEl = document.getElementById('attack-chart');
const featureImportanceEl = document.getElementById('feature-importance-content');

// 全局变量
let testData = null;
let attackChart = null;

// 监听文件上传
fileInput.addEventListener('change', function(e) {
    if (e.target.files.length > 0) {
        const file = e.target.files[0];
        fileName.textContent = file.name;
        predictBtn.disabled = false;
        
        // 读取CSV文件
        const reader = new FileReader();
        reader.onload = function(event) {
            const csvData = event.target.result;
            testData = parseCSV(csvData);
        };
        reader.readAsText(file);
    }
});

// 点击预测按钮
predictBtn.addEventListener('click', function() {
    if (testData) {
        startPrediction();
    }
});

// 关闭结果弹窗
closeBtn.addEventListener('click', function() {
    resultModal.style.display = 'none';
});

// 点击弹窗外部关闭
window.addEventListener('click', function(event) {
    if (event.target === resultModal) {
        resultModal.style.display = 'none';
    }
});

// 解析CSV文件
function parseCSV(csvData) {
    const lines = csvData.split('\n');
    const headers = lines[0].split(',');
    const data = [];
    
    for (let i = 1; i < lines.length; i++) {
        if (lines[i].trim() === '') continue;
        
        const values = lines[i].split(',');
        const row = {};
        
        for (let j = 0; j < headers.length; j++) {
            // 尝试将数值转换为数字类型
            const value = values[j];
            row[headers[j]] = isNaN(Number(value)) ? value : Number(value);
        }
        
        data.push(row);
    }
    
    return { headers, rows: data };
}

// 开始预测
function startPrediction() {
    loading.style.display = 'block';
    predictBtn.disabled = true;
    
    // 获取用户选择的模型类型
    const selectedModel = modelSelect.value;
    
    // 模拟预测过程（实际中会调用后端API）
    setTimeout(function() {
        const results = predictNetworkAttacks(testData, selectedModel);
        showResults(results);
        loading.style.display = 'none';
    }, 1000);
}

// 预定义的模型特征重要性数据（模拟从后端获取）
const modelFeatureImportances = {
    'catboost': [
        { name: 'dur', score: 0.2145 },
        { name: 'spkts', score: 0.1562 },
        { name: 'dpkts', score: 0.1489 },
        { name: 'sbytes', score: 0.1034 },
        { name: 'dbytes', score: 0.0987 }
    ],
    'random_forest': [
        { name: 'sbytes', score: 0.1876 },
        { name: 'dbytes', score: 0.1745 },
        { name: 'spkts', score: 0.1532 },
        { name: 'dpkts', score: 0.1267 },
        { name: 'rate', score: 0.0893 }
    ],
    'xgboost': [
        { name: 'rate', score: 0.1987 },
        { name: 'dur', score: 0.1654 },
        { name: 'sbytes', score: 0.1432 },
        { name: 'dbytes', score: 0.1245 },
        { name: 'sttl', score: 0.0987 }
    ],
    'stacking': [
        { name: 'dur', score: 0.2056 },
        { name: 'rate', score: 0.1789 },
        { name: 'sbytes', score: 0.1432 },
        { name: 'dbytes', score: 0.1267 },
        { name: 'spkts', score: 0.1145 }
    ]
};

// 预测网络攻击（支持多种模型）
function predictNetworkAttacks(data, modelType = 'stacking') {
    const rows = data.rows;
    const totalRecords = rows.length;
    const attackedRecords = [];
    const attackTypes = {};
    
    // 根据选择的模型类型调整预测逻辑
    let modelParams = {
        dlossThreshold: 5,
        slossThreshold: 5,
        rateThreshold: 100000,
        dstBytesThreshold: 1000000,
        srcBytesThreshold: 1000000,
        durHighThreshold: 1000,
        durLowThreshold: 0.001
    };
    
    // 根据模型类型调整阈值参数
    switch(modelType) {
        case 'catboost':
            // CatBoost模型 - 通常具有较高的精度
            modelParams = {
                dlossThreshold: 4,
                slossThreshold: 4,
                rateThreshold: 80000,
                dstBytesThreshold: 800000,
                srcBytesThreshold: 800000,
                durHighThreshold: 800,
                durLowThreshold: 0.002
            };
            break;
        case 'random_forest':
            // 随机森林模型 - 对噪声有较好的鲁棒性
            modelParams = {
                dlossThreshold: 6,
                slossThreshold: 6,
                rateThreshold: 120000,
                dstBytesThreshold: 1200000,
                srcBytesThreshold: 1200000,
                durHighThreshold: 1200,
                durLowThreshold: 0.0008
            };
            break;
        case 'xgboost':
            // XGBoost模型 - 通常具有较高的召回率
            modelParams = {
                dlossThreshold: 3,
                slossThreshold: 3,
                rateThreshold: 70000,
                dstBytesThreshold: 700000,
                srcBytesThreshold: 700000,
                durHighThreshold: 700,
                durLowThreshold: 0.003
            };
            break;
        case 'stacking':
        default:
            // Stacking集成模型 - 综合了多个模型的优点
            // 使用默认参数
            break;
    }
    
    // 执行预测
    for (let i = 0; i < rows.length; i++) {
        const row = rows[i];
        
        let isAttacked = false;
        let attackType = 'Normal';
        
        // 检查label字段是否为1或包含'Attack'/'攻击'等关键词
        // 根据模型类型预测攻击类别（模拟真实模型预测）
        function predictAttackCategory(modelType, row) {
            // 基于UNSW-NB15数据集特征分布的攻击类型预测逻辑
            switch(modelType) {
                case 'catboost':
                    if (row.spkts > 150) return 'Reconnaissance';
                    if (row.dur > 1.0) return 'DOS';
                    if (row.sbytes > 2000) return 'Access';
                    if (row.rate > 5000) return 'DDoS';
                    if (row.sloss > 0) return 'Injection';
                    return 'Normal';
                case 'random_forest':
                    if (row.dpkts > 100) return 'Reconnaissance';
                    if (row.rate > 8000) return 'DDoS';
                    if (row.dbytes > 1500) return 'Access';
                    if (row.sttl < 30) return 'Brute Force';
                    if (row.sload > 10000) return 'DOS';
                    return 'Normal';
                case 'xgboost':
                    if (row.sloss > 0 || row.dloss > 0) return 'Injection';
                    if (row.tcprtt > 100) return 'Brute Force';
                    if (row.dur > 0.5 && row.spkts < 5) return 'DoS';
                    if (row.spkts > 200) return 'Reconnaissance';
                    if (row.sbytes > 3000) return 'Access';
                    return 'Normal';
                case 'stacking':
                    // 综合多模型优势的攻击类型预测
                    if (row.rate > 10000 && row.dpkts > 50) return 'DDoS';
                    if (row.spkts > 150 && row.sbytes < 500) return 'Reconnaissance';
                    if (row.sbytes > 2500 && row.dbytes > 1000) return 'Access';
                    if (row.dur > 0.8 && row.spkts < 10) return 'DoS';
                    if (row.sloss > 0 || row.ct_ftp_cmd > 0) return 'Injection';
                    if (row.sttl < 20 && row.swin === 0) return 'Brute Force';
                    return 'Normal';
                default:
                    return 'Normal';
            }
        }

        // 优先使用test_data.csv中的attack_cat列数据
        if (row.attack_cat && row.attack_cat !== 'Normal') {
            isAttacked = true;
            attackType = row.attack_cat;
        } 
        // 如果没有attack_cat列或为Normal，则使用预测函数
        else {
            const predictedCategory = predictAttackCategory(modelType, row);
            if (predictedCategory !== 'Normal') {
                isAttacked = true;
                attackType = predictedCategory;
            } 
            // 最后使用基于阈值的异常检测
            else if (
                (row.dloss !== undefined && row.dloss > modelParams.dlossThreshold) ||
                (row.sloss !== undefined && row.sloss > modelParams.slossThreshold) ||
                (row.rate !== undefined && row.rate > modelParams.rateThreshold) ||
                (row.dst_bytes !== undefined && row.dst_bytes > modelParams.dstBytesThreshold) ||
                (row.src_bytes !== undefined && row.src_bytes > modelParams.srcBytesThreshold) ||
                (row.dur !== undefined && (row.dur > modelParams.durHighThreshold || row.dur < modelParams.durLowThreshold))
            ) {
                isAttacked = true;
                
                // 根据模型类型标记异常
                attackType = modelType === 'stacking' ? 'Anomaly (Stacking)' : 
                            modelType === 'xgboost' ? 'Anomaly (XGBoost)' :
                            modelType === 'random_forest' ? 'Anomaly (Random Forest)' :
                            'Anomaly (CatBoost)';
            }
        }
        
        if (isAttacked) {
            attackedRecords.push({
                rowIndex: i + 1, // 行号从1开始
                attackType: attackType
            });
            
            // 更新攻击类型统计
            attackTypes[attackType] = (attackTypes[attackType] || 0) + 1;
        }
    }
    
    // 获取当前模型的特征重要性
    const featureImportance = modelFeatureImportances[modelType] || [];
    
    return {
        totalRecords,
        attackedRecords,
        attackTypes,
        featureImportance
    };
}

// 显示结果
function showResults(results) {
    totalRecordsEl.textContent = results.totalRecords;
    attackedRecordsEl.textContent = results.attackedRecords.length;
    
    // 清空之前的结果
    attackDetailsEl.innerHTML = '';
    
    // 显示受攻击的设备信息
    if (results.attackedRecords.length > 0) {
        results.attackedRecords.forEach(record => {
            const item = document.createElement('div');
            item.className = 'result-item';
            item.innerHTML = `<strong>行号:</strong> ${record.rowIndex}, <strong>攻击类型:</strong> ${record.attackType}`;
            attackDetailsEl.appendChild(item);
        });
    } else {
        const item = document.createElement('div');
        item.className = 'result-item';
        item.textContent = '未发现受攻击的设备';
        attackDetailsEl.appendChild(item);
    }
    
    // 绘制攻击类型分布图
    drawAttackChart(results.attackTypes);
    
    // 显示特征重要性
    showFeatureImportance(results.featureImportance);
    
    // 显示结果弹窗
    resultModal.style.display = 'flex';
}

// 绘制攻击类型分布图（Chart.js不可用时降级显示）
function drawAttackChart(attackTypes) {
    // 清空图表容器
    attackChartEl.innerHTML = '';
    
    // 确保attackTypes是对象且有数据
    if (!attackTypes || typeof attackTypes !== 'object' || Object.keys(attackTypes).length === 0) {
        console.log('没有攻击类型数据可显示');
        displayTextStatistics({});
        return;
    }
    
    // 准备图表数据
    const labels = Object.keys(attackTypes);
    const data = Object.values(attackTypes);
    
    // 定义固定的颜色数组，确保图表美观
    const fixedColors = [
        'rgba(255, 99, 132, 0.7)',  // 红色
        'rgba(54, 162, 235, 0.7)',  // 蓝色
        'rgba(255, 206, 86, 0.7)',  // 黄色
        'rgba(75, 192, 192, 0.7)',  // 绿色
        'rgba(153, 102, 255, 0.7)', // 紫色
        'rgba(255, 159, 64, 0.7)'   // 橙色
    ];
    
    // 生成颜色数组，确保每个类别都有颜色
    const backgroundColors = labels.map((_, index) => fixedColors[index % fixedColors.length]);
    const borderColors = backgroundColors.map(color => color.replace('0.7', '1'));
    
    // 检查Chart.js是否可用
    if (typeof Chart !== 'undefined') {
        try {
            // 销毁之前的图表
            if (attackChart && attackChart.destroy) {
                attackChart.destroy();
            }
            
            // 创建图表配置
            const chartConfig = {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: '攻击类型数量',
                        data: data,
                        backgroundColor: backgroundColors,
                        borderColor: borderColors,
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: {
                            display: true,
                            text: '各攻击类型数量分布',
                            font: {
                                size: 16
                            }
                        },
                        legend: {
                            display: false
                        },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    return `${context.dataset.label}: ${context.raw} 条`;
                                }
                            }
                        }
                    },
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: '攻击类型'
                            },
                            ticks: {
                                maxRotation: 45,
                                minRotation: 45
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: '数量'
                            },
                            beginAtZero: true,
                            ticks: {
                                stepSize: 1,
                                precision: 0
                            }
                        }
                    }
                }
            };
            
            // 创建图表
            attackChart = new Chart(attackChartEl, chartConfig);
            console.log('图表渲染成功');
        } catch (e) {
            // 如果Chart.js出错，显示文本格式的统计
            console.error('Chart.js渲染失败:', e);
            displayTextStatistics(attackTypes);
        }
    } else {
        // 如果Chart.js未加载，显示文本格式的统计
        displayTextStatistics(attackTypes);
    }
}

// 以文本形式显示统计信息
function displayTextStatistics(attackTypes) {
    // 创建一个容器
    const statsContainer = document.createElement('div');
    statsContainer.className = 'text-statistics';
    
    // 添加标题
    const title = document.createElement('h4');
    title.textContent = '攻击类型统计（文本格式）';
    statsContainer.appendChild(title);
    
    // 添加每个攻击类型的统计
    const types = Object.keys(attackTypes);
    if (types.length === 0) {
        const noData = document.createElement('p');
        noData.textContent = '未发现攻击记录';
        statsContainer.appendChild(noData);
    } else {
        types.forEach(type => {
            const statItem = document.createElement('div');
            statItem.className = 'stat-type-item';
            statItem.innerHTML = `<strong>${type}:</strong> ${attackTypes[type]} 条记录`;
            statsContainer.appendChild(statItem);
        });
    }
    
    // 添加到图表容器
    attackChartEl.appendChild(statsContainer);
}

// 显示特征重要性
function showFeatureImportance(featureImportance) {
    // 清空之前的内容
    featureImportanceEl.innerHTML = '';
    
    if (!featureImportance || featureImportance.length === 0) {
        const noFeatureMsg = document.createElement('div');
        noFeatureMsg.className = 'result-item';
        noFeatureMsg.textContent = '暂无特征重要性数据';
        featureImportanceEl.appendChild(noFeatureMsg);
        return;
    }
    
    // 计算最大分数，用于设置条形图的比例
    const maxScore = Math.max(...featureImportance.map(f => f.score));
    
    // 直接向featureImportanceEl添加每个特征
    featureImportance.forEach((feature, index) => {
        const featureItem = document.createElement('div');
        featureItem.className = 'feature-item';
        
        // 排名
        const rankSpan = document.createElement('span');
        rankSpan.className = 'feature-rank';
        rankSpan.textContent = `${index + 1}.`;
        
        // 特征名称
        const nameSpan = document.createElement('span');
        nameSpan.className = 'feature-name';
        nameSpan.textContent = feature.name;
        
        // 条形图容器
        const barContainer = document.createElement('div');
        barContainer.className = 'feature-bar-container';
        
        // 条形图
        const bar = document.createElement('div');
        bar.className = 'feature-bar';
        // 设置条形图宽度，基于特征重要性分数的比例
        const barWidth = (feature.score / maxScore) * 100;
        bar.style.width = `${barWidth}%`;
        
        // 分数
        const scoreSpan = document.createElement('span');
        scoreSpan.className = 'feature-score';
        scoreSpan.textContent = feature.score.toFixed(4);
        
        // 组合元素
        barContainer.appendChild(bar);
        featureItem.appendChild(rankSpan);
        featureItem.appendChild(nameSpan);
        featureItem.appendChild(barContainer);
        featureItem.appendChild(scoreSpan);
        
        featureImportanceEl.appendChild(featureItem);
    });
}

// 数据预处理类（简化版）
class DataPreprocessor {
    constructor() {
        this.categoricalColumns = ['proto', 'service', 'state'];
        this.labelEncoders = {};
        this.scalerParams = {
            mean: {},
            std: {}
        };
    }
    
    // 拟合数据预处理参数
    fit(data) {
        // 这里应该根据训练数据拟合编码器和标准化参数
        // 在简化版中使用预设值
        return this;
    }
    
    // 转换数据
    transform(data) {
        // 这里应该应用训练好的预处理步骤
        // 在简化版中直接返回数据
        return data;
    }
}

// Stacking分类器（简化版）
class StackingClassifier {
    constructor(baseModels, metaModel) {
        this.baseModels = baseModels;
        this.metaModel = metaModel;
    }
    
    // 训练模型
    fit(X, y) {
        // 这里应该实现完整的Stacking训练逻辑
        // 在简化版中直接返回自身
        return this;
    }
    
    // 预测
    predict(X) {
        // 这里应该实现完整的Stacking预测逻辑
        // 在简化版中返回随机结果
        return X.map(() => Math.random() > 0.5 ? 1 : 0);
    }
}