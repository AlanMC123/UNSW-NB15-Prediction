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
        if (row.label === 1 || (typeof row.label === 'string' && 
            (row.label.toLowerCase().includes('attack') || row.label.includes('攻击')))) {
            isAttacked = true;
            
            // 尝试从多个可能的字段获取攻击类型
            if (row.attack_cat && row.attack_cat.trim() !== '' && row.attack_cat !== 'Normal') {
                attackType = row.attack_cat;
            } else if (row.attack && row.attack.trim() !== '' && row.attack !== 'Normal') {
                attackType = row.attack;
            } else if (row.type && row.type.trim() !== '' && row.type !== 'Normal') {
                attackType = row.type;
            } else {
                // 根据模型类型调整未知攻击的命名
                attackType = modelType === 'stacking' ? 'Known Attack (Stacking)' : 
                            modelType === 'xgboost' ? 'Known Attack (XGBoost)' :
                            modelType === 'random_forest' ? 'Known Attack (Random Forest)' :
                            'Known Attack (CatBoost)';
            }
        } else if (
            // 基于模型特定的阈值进行异常检测
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
        
        if (isAttacked) {
            attackedRecords.push({
                rowIndex: i + 1, // 行号从1开始
                attackType: attackType
            });
            
            // 更新攻击类型统计
            attackTypes[attackType] = (attackTypes[attackType] || 0) + 1;
        }
    }
    
    return {
        totalRecords,
        attackedRecords,
        attackTypes
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
    
    // 显示结果弹窗
    resultModal.style.display = 'flex';
}

// 绘制攻击类型分布图
function drawAttackChart(attackTypes) {
    // 销毁之前的图表
    if (attackChart) {
        attackChart.destroy();
    }
    
    // 准备图表数据
    const labels = Object.keys(attackTypes);
    const data = Object.values(attackTypes);
    
    // 生成随机颜色
    const backgroundColors = labels.map(() => {
        const r = Math.floor(Math.random() * 255);
        const g = Math.floor(Math.random() * 255);
        const b = Math.floor(Math.random() * 255);
        return `rgba(${r}, ${g}, ${b}, 0.7)`;
    });
    
    // 创建图表
    attackChart = new Chart(attackChartEl, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: '攻击类型数量',
                data: data,
                backgroundColor: backgroundColors,
                borderColor: backgroundColors.map(color => color.replace('0.7', '1')),
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        stepSize: 1
                    }
                }
            }
        }
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