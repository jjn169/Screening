import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sksurv.util import Surv

# 加载保存的CoxPHSurvivalAnalysis模型
model_path = r"F:\小论文8\机器学习2\模型1\GBM_Surv.joblib"
cox_model = joblib.load(model_path)

# 加载数据并处理
data_path = r"F:\小论文8\机器学习2\数据\2.xlsx"
df = pd.read_excel(data_path)

# 对 race 变量进行独热编码
df = pd.get_dummies(df, columns=['race'], drop_first=True)

# 模型训练时使用的所有特征
all_features_used_in_training = [
    'edu', 'marry', 'PIR', 'job', 'insurance', 'Foodsecurity', 'Healthcare', 'housing',
    'BMIclass', 'sex', 'age', 'race_Other hispanics', 'race_Non-hispanic whites',
    'race_Non-hispanic blacks', 'race_The other RACES'
]

# 你感兴趣的特征
features_to_visualize = [
    'edu', 'marry', 'PIR', 'job', 'insurance', 'Foodsecurity', 'Healthcare', 'housing'
]

# 不感兴趣的特征
features_to_control = list(set(all_features_used_in_training) - set(features_to_visualize))

# 检查是否所有训练时的特征都存在于数据中
missing_features = set(all_features_used_in_training) - set(df.columns)
if missing_features:
    raise ValueError(f"缺少以下特征: {missing_features}")

# 使用所有训练时的特征进行预测
X = df[all_features_used_in_training].copy()

# 保持不感兴趣的特征固定不变（使用其平均值或中位数）
for feature in features_to_control:
    X[feature] = X[feature].mean()

# 将数据转换为 NumPy 数组以消除特征名称警告
X_array = X.values

# 构造生存对象
y = Surv.from_dataframe('Status', 'Time', df)

# 计算累积风险函数
cumulative_hazard_funcs = cox_model.predict_cumulative_hazard_function(X_array)

# 可视化累积风险，不添加标签
plt.figure(figsize=(10, 8))
for i, chf in enumerate(cumulative_hazard_funcs):
    if i < 10:  # 仅显示前10个样本的曲线
        plt.step(chf.x, chf.y, where="post", alpha=0.5)  # 不添加 label 参数

plt.xlabel("Time")
plt.ylabel("Cumulative Hazard")
plt.title("Cumulative Hazard Function - CoxPHSurvivalAnalysis")
plt.grid(True)

# 保存为 jpg 格式，300 DPI
output_path = r"F:\小论文8\机器学习2\特征\cumulative_hazard.jpg"
plt.savefig(output_path, format="jpg", dpi=300)

plt.show()

print(f"Cumulative Hazard plot saved to {output_path}")
