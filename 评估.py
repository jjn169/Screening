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
    
    # 选择与结局变量为1的SHAP值
    shap_values_selected = shap_values[..., 1]

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

# 检查数据特征数量
if data.shape[1] == 16:
    st.write("数据中包含所有16个特征。")

# 显示所有特征的SHAP Summary Plot
shap.summary_plot(shap_values_selected, data, max_display=16)
