import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from joblib import load
from sklearn.metrics import confusion_matrix

# 读取数据
data_path = "F:\\小论文8\\机器学习\\数据\\分析2.xlsx"
data = pd.read_excel(data_path)

# 独热编码race变量
encoder = OneHotEncoder(drop='first')
encoded_race = encoder.fit_transform(data[['race']]).toarray()
encoded_race_df = pd.DataFrame(encoded_race, columns=encoder.get_feature_names_out(['race']))
data = pd.concat([data.drop(columns=['race']), encoded_race_df], axis=1)

# 提取特征和标签
X = data.drop(columns=['cancer'])
y = data['cancer']

# 加载模型
model_paths = {
    "LightGBM": "F:\\小论文8\\机器学习\\模型1\\LightGBM.joblib",
    "LogisticRegression": "F:\\小论文8\\机器学习\\模型1\\LogisticRegression.joblib",
    "RF": "F:\\小论文8\\机器学习\\模型1\\RF.joblib",
    "SVM": "F:\\小论文8\\机器学习\\模型1\\SVM.joblib",
    "XGBoost": "F:\\小论文8\\机器学习\\模型1\\XGBoost.joblib",
    "MLP": "F:\\小论文8\\机器学习\\模型1\\MLP.joblib",
    "KNN": "F:\\小论文8\\机器学习\\模型1\\KNN.joblib",
    "AdaBoost": "F:\\小论文8\\机器学习\\模型1\\AdaBoost.joblib",
    "GradientBoosting": "F:\\小论文8\\机器学习\\模型1\\GradientBoosting.joblib"
}

models = {name: load(path) for name, path in model_paths.items()}


# 计算净效益的函数
def calculate_net_benefit_model(thresh_group, y_pred_score, y_label):
    net_benefit_model = np.array([])
    for thresh in thresh_group:
        y_pred_label = y_pred_score > thresh
        tn, fp, fn, tp = confusion_matrix(y_label, y_pred_label).ravel()
        n = len(y_label)
        net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
        net_benefit_model = np.append(net_benefit_model, net_benefit)
    return net_benefit_model


def calculate_net_benefit_all(thresh_group, y_label):
    net_benefit_all = np.array([])
    tn, fp, fn, tp = confusion_matrix(y_label, y_label).ravel()
    total = tp + tn
    for thresh in thresh_group:
        net_benefit = (tp / total) - (tn / total) * (thresh / (1 - thresh))
        net_benefit_all = np.append(net_benefit_all, net_benefit)
    return net_benefit_all


def plot_DCA(ax, thresh_group, net_benefit_model, net_benefit_all, model_name):
    # Plot
    ax.plot(thresh_group, net_benefit_model, label=f'{model_name}')

    # Fill，显示出模型较于treat all和treat none好的部分
    y2 = np.maximum(net_benefit_all, 0)
    y1 = np.maximum(net_benefit_model, y2)
    ax.fill_between(thresh_group, y1, y2, alpha=0.2)

    return ax
# 预测并计算净效益
thresh_group = np.arange(0, 1, 0.01)
fig, ax = plt.subplots(figsize=(10, 8))

# 计算Treat all和Treat none的净效益
net_benefit_all = calculate_net_benefit_all(thresh_group, y)
ax.plot(thresh_group, net_benefit_all, color='black', label='Treat all')
ax.plot((0, 1), (0, 0), color='black', linestyle=':', label='Treat none')

for model_name, model in models.items():
    try:
        y_pred_score = model.predict_proba(X)[:, 1]
    except AttributeError:
        print(f"{model_name} does not support predict_proba, skipping.")
        continue
    net_benefit_model = calculate_net_benefit_model(thresh_group, y_pred_score, y)
    ax = plot_DCA(ax, thresh_group, net_benefit_model, net_benefit_all, model_name)

# 调整纵坐标刻度
ax.set_yticks(np.arange(-0.2, 0.4, 0.1))

# 添加图例和标签
ax.set_xlim(0, 1)
ax.set_ylim(-0.2, 0.4)
ax.set_xlabel('Threshold Probability', fontdict={'family': 'Times New Roman', 'fontsize': 15})
ax.set_ylabel('Net Benefit', fontdict={'family': 'Times New Roman', 'fontsize': 15})
ax.legend(loc='upper right')
ax.grid('major')
ax.spines['right'].set_color((0.8, 0.8, 0.8))
ax.spines['top'].set_color((0.8, 0.8, 0.8))

# 展示图像
plt.show()
