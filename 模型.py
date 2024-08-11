import joblib

# 模型路径
model_paths = {
    'LightGBM': 'F:\\小论文8\\机器学习\\模型1\\LightGBM.joblib',
    'LogisticRegression': 'F:\\小论文8\\机器学习\\模型1\\LogisticRegression.joblib',
    'RF': 'F:\\小论文8\\机器学习\\模型1\\RF.joblib',
    'SVM': 'F:\\小论文8\\机器学习\\模型1\\SVM.joblib',
    'XGBoost': 'F:\\小论文8\\机器学习\\模型1\\XGBoost.joblib',
    'MLP': 'F:\\小论文8\\机器学习\\模型1\\MLP.joblib',
    'DecisionTree': 'F:\\小论文8\\机器学习\\模型1\\DecisionTree.joblib',
    'KNN': 'F:\\小论文8\\机器学习\\模型1\\KNN.joblib',
    'AdaBoost': 'F:\\小论文8\\机器学习\\模型1\\AdaBoost.joblib',
    'GradientBoosting': 'F:\\小论文8\\机器学习\\模型1\\GradientBoosting.joblib'
}

models = {name: joblib.load(path) for name, path in model_paths.items()}
