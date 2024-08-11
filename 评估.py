import streamlit as st
import shap
import matplotlib.pyplot as plt
import numpy as np
import joblib
import pandas as pd

# 加载SHAP值
shap_values = np.load(r"F:\小论文8\机器学习\sha\随机森林\shap1.npy", allow_pickle=True)

# 加载模型
model = joblib.load(r"F:\小论文8\机器学习\模型2\RandomForest.joblib")

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
