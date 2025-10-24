import pandas as pd
from tqdm import tqdm
import chardet  # 自动检测编码

# ======= 文件路径设置 =======
feature_file = "NUSW-NB15_features.csv"   # 特征说明文件
data_file = "UNSW_NB15_training-set.csv"               # 数据集文件

# ======= 1️⃣ 自动检测特征说明文件编码 =======
with open(feature_file, "rb") as f:
    detect_result = chardet.detect(f.read(2048))
encoding = detect_result["encoding"]
print(f"📘 检测到特征说明文件编码：{encoding}")

# 重新读取文件
try:
    features = pd.read_csv(feature_file, encoding=encoding)
except Exception as e:
    print("❌ 无法读取特征说明文件：", e)
    exit()

expected_cols = {"No.", "Name", "Type", "Description"}

if not expected_cols.issubset(features.columns):
    print("⚠️ 特征说明文件的列名与预期不符，实际列名为：", list(features.columns))
    exit()

# ======= 根据 Type 列判断变量类型 =======
type_map = {
    "binary": "0/1 二分变量",
    "nominal": "分类变量 (非0/1)",
    "integer": "数值变量",
    "float": "数值变量",
    "numeric": "数值变量"
}
features["变量类别"] = features["Type"].apply(
    lambda x: type_map.get(str(x).lower(), "其他类型")
)

binary_vars = features.loc[features["变量类别"] == "0/1 二分变量", "Name"].tolist()

print(f"✅ 识别到 {len(binary_vars)} 个 0/1 二分变量：")
print(binary_vars)

# ======= 2️⃣ 读取真实数据集 =======
try:
    df = pd.read_csv(data_file)
except Exception as e:
    print("❌ 无法读取数据集文件：", e)
    exit()

print(f"\n📊 原始数据集：{df.shape[0]} 行 × {df.shape[1]} 列")

# ======= 3️⃣ 清理非法 0/1 值（带进度条） =======
invalid_rows = 0
print("\n🚧 正在检查并删除 0/1 变量中非法取值...")
for col in tqdm(binary_vars, desc="检查0/1变量", ncols=80):
    if col in df.columns:
        invalid_mask = ~df[col].isin([0, 1])
        count_invalid = invalid_mask.sum()
        if count_invalid > 0:
            invalid_rows += count_invalid
            df = df[~invalid_mask]

if invalid_rows == 0:
    print("✅ 所有0/1变量的值均合法。")
else:
    print(f"⚠️ 共删除 {invalid_rows} 行包含非法0/1值的数据。")

# ======= 4️⃣ 去除特征冲突行（忽略 id 列） =======
print("\n🚧 正在检查并删除特征冲突行（特征相同但 label 或 attack_cat 不同）...")

# 忽略 id、label、attack_cat 三个列
ignore_cols = ["id", "attack_cat", "label"]
non_label_cols = [c for c in df.columns if c not in ignore_cols]

# 分组检测
conflict_idx = []

# tqdm 不直接支持 groupby，这里显示总体进度提示
grouped = df.groupby(non_label_cols, dropna=False)
for _, group in tqdm(grouped, total=len(grouped), desc="检查冲突", ncols=80):
    # 若同一组中 label 或 attack_cat 有多个不同值 → 冲突
    if len(group[["attack_cat", "label"]].drop_duplicates()) > 1:
        conflict_idx.extend(group.index)

# 删除冲突行
if conflict_idx:
    conflict_count = len(conflict_idx)
    df = df.drop(conflict_idx)
    print(f"⚠️ 检测到 {conflict_count} 行特征冲突数据（忽略 id），已删除。")
else:
    print("✅ 未检测到特征冲突行。")

# ======= 5️⃣ 输出统计与保存 =======
print(f"\n📊 清理后数据集：{df.shape[0]} 行 × {df.shape[1]} 列")

output_file = "UNSW-NB15_cleaned_tr.csv"
df.to_csv(output_file, index=False, encoding="utf-8-sig")
print(f"✅ 已保存清理后的数据集：{output_file}")

# ======= 附加：输出类型统计 =======
print("\n===== 特征类型统计 =====")
print(features["变量类别"].value_counts())
