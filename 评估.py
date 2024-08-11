import requests
import streamlit as st
import shap
import matplotlib.pyplot as plt
import numpy as np
import joblib
import pandas as pd
GITHUB_USER = 'jjn169'
REPO_NAME = 'Screening'
TAG_NAME = 'rf'  # 这是你的Tag名称
FILE_NAME = 'RandomForest.joblib'

# 构建GitHub文件下载URL
url = f'https://github.com/{GITHUB_USER}/{REPO_NAME}/raw/{TAG_NAME}/{FILE_NAME}'

# 下载文件
response = requests.get(url)
if response.status_code == 200:
    # 将文件保存到本地临时目录
    with open(FILE_NAME, 'wb') as f:
        f.write(response.content)
    # 加载模型文件
    model = joblib.load(FILE_NAME)
    os.remove(FILE_NAME)  # 加载后删除文件
else:
    print(f"Failed to download file: {response.status_code}")
# 加载SHAP值
shap_values = np.load("shap1.npy", allow_pickle=True)


# 读取数据（假设与训练模型时相同的数据）
data = pd.read_excel(r"F:\小论文8\机器学习2\数据\2.xlsx")

# 处理race变量进行独热编码
data = pd.get_dummies(data, columns=["race"])

# 检查SHAP值和数据的形状
st.write(f"SHAP values shape: {shap_values.shape}")
st.write(f"Data shape: {data.shape}")

# 展示SHAP Summary Plot
if shap_values.shape[1] == data.shape[1]:
    st.header("SHAP Summary Plot")
    plt.figure()
    shap.summary_plot(shap_values, data)
    st.pyplot(plt)
else:
    st.error("SHAP值和数据的形状不匹配，请检查数据处理过程。")
