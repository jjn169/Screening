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
    model = joblib.load(MODEL_FILE_NAME)
    os.remove(MODEL_FILE_NAME)
    st.success("模型文件加载成功。")
except requests.exceptions.RequestException as e:
    st.error(f"无法下载模型文件: {e}")
    st.stop()
except Exception as e:
    st.error(f"加载模型文件时出错: {e}")
    st.stop()

# 加载SHAP值
try:
    shap_values = np.load(SHAP_FILE_NAME, allow_pickle=True)
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
