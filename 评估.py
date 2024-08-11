import requests
import streamlit as st
import shap
import matplotlib.pyplot as plt
import numpy as np
import joblib
import pandas as pd
import os

# GitHub 仓库信息
GITHUB_USER = 'jjn169'
REPO_NAME = 'Screening'
TAG_NAME = 'rf'
MODEL_FILE_NAME = 'RandomForest.joblib'
DATA_FILE_NAME = '分析2.xlsx'
SHAP_FILE_NAME = 'shap1.npy'

# 使用 Release 下载的URL
model_url = f'https://github.com/{GITHUB_USER}/{REPO_NAME}/releases/download/{TAG_NAME}/{MODEL_FILE_NAME}'
data_url = f'https://github.com/{GITHUB_USER}/{REPO_NAME}/raw/main/{DATA_FILE_NAME}'

# 下载并加载模型文件
try:
    response = requests.get(model_url)
    response.raise_for_status()
    with open(MODEL_FILE_NAME, 'wb') as f:
        f.write(response.content)
    
    if os.path.exists(MODEL_FILE_NAME):
        model = joblib.load(MODEL_FILE_NAME)
        os.remove(MODEL_FILE_NAME)
        st.success("模型文件加载成功。")
    else:
        st.error(f"模型文件 '{MODEL_FILE_NAME}' 未找到，请检查文件下载是否成功。")
        st.stop()

except requests.exceptions.RequestException as e:
    st.error(f"无法下载模型文件: {e}")
    st.stop()
except Exception as e:
    st.error(f"加载模型文件时出错: {e}")
    st.stop()

# 加载SHAP值
try:
    shap_values = np.load(SHAP_FILE_NAME, allow_pickle=True)
    st.write(f"SHAP values shape: {shap_values.shape}")
except FileNotFoundError:
    st.error(f"未找到SHAP文件：{SHAP_FILE_NAME}")
    st.stop()
except Exception as e:
    st.error(f"加载SHAP值时出错: {e}")
    st.stop()

# 下载并加载数据文件
try:
    response = requests.get(data_url)
    response.raise_for_status()
    with open(DATA_FILE_NAME, 'wb') as f:
        f.write(response.content)
    data = pd.read_excel(DATA_FILE_NAME)
    os.remove(DATA_FILE_NAME)
    st.success("数据文件加载成功。")
except requests.exceptions.RequestException as e:
    st.error(f"无法下载数据文件: {e}")
    st.stop()
except Exception as e:
    st.error(f"加载数据文件时出错: {e}")
    st.stop()

# 处理 race 变量进行独热编码
try:
    data = pd.get_dummies(data, columns=["race"])
except KeyError as e:
    st.error(f"数据中缺少 'race' 列: {e}")
    st.stop()

# 选择与结局变量为1对应的SHAP值（假设第三个维度中索引1对应结局变量1）
shap_values_selected = shap_values[..., 1]

# 检查数据和SHAP值的特征匹配
if shap_values_selected.shape[1] != data.shape[1]:
    st.warning("SHAP值的特征数量与数据特征数量不匹配。尝试对齐特征...")
    
    if shap_values_selected.shape[1] < data.shape[1]:
        st.warning("数据特征列多于SHAP值特征列，选择前部分特征...")
        data = data.iloc[:, :shap_values_selected.shape[1]]  # 只保留前 N 个特征，N 是SHAP值的特征数量
    else:
        st.error("SHAP值的特征数量多于数据特征，这可能不正确。")
        st.stop()

# 展示SHAP Summary Plot
st.write(f"Data shape after alignment: {data.shape}")
if shap_values_selected.shape[1] == data.shape[1]:
    st.header("SHAP Summary Plot")
    plt.figure()
    shap.summary_plot(shap_values_selected, data)
    st.pyplot(plt)
else:
    st.error("SHAP值和数据的形状仍然不匹配，请检查数据处理过程。")
