# Using Gradient Descent (GD) to Solve Linear Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# read the CSV file
df = pd.read_csv('lifespan_data_full.csv')

# show the first 5 rows of the dataframe
print(df.head())

# extract the features and target variable
X0 = df[['parent_lifespan', 'gender', 'exercise_hours', \
        'smoking', 'diet_health', 'sleep_hours', 'stress_level']].values
y0 = df['actual_lifespan'].values

# use the first 900 samples for training (100 samples are left for model evaluation)
X = X0[0:900]
y = y0[0:900]

# ===================== 第三步：梯度下降法求解权重 =====================
def gradient_descent(X, y, learning_rate=0.0001, epochs=150000, tol=1e-6):
    """
    梯度下降法求解线性回归参数
    参数：
        X：未添加截距项的特征矩阵
        y：因变量
        learning_rate：学习率
        epochs：最大迭代次数
        tol：收敛阈值
    返回：截距b，权重w，损失历史
    """
    n_samples, n_features = X.shape
    # 初始化权重（随机正态分布）
    w = np.random.randn(n_features)
    b = np.random.randn()
    loss_history = []  # 记录损失变化
    
    for epoch in range(epochs):
        # 前向预测：y_pred = b + X·w
        y_pred = b + np.dot(X, w)
        # 计算均方误差损失
        loss = np.mean((y_pred - y) ** 2)
        loss_history.append(loss)
        
        # 计算梯度
        dw = (2 / n_samples) * np.dot(X.T, (y_pred - y))  # 权重梯度
        db = (2 / n_samples) * np.sum(y_pred - y)         # 截距梯度
        
        # 更新参数
        w -= learning_rate * dw
        b -= learning_rate * db
        
        # 收敛判断
        if epoch > 0 and abs(loss_history[-1] - loss_history[-2]) < tol:
            print(f"\n梯度下降在第{epoch}轮收敛！")
            break
    
    return b, w, loss_history

# 求解梯度下降解
b, w, loss_history = gradient_descent(X, y, learning_rate=0.0001, epochs=150000)
print("\n===== Linear Regression Solution =====")
print(f"Intercept (bias): {b:.4f}")
print(f"Weights: {np.round(w, 4)}")

# 绘制梯度下降损失曲线（科普可视化）
plt.figure(figsize=(10, 4))
plt.plot(loss_history, color='blue', linewidth=1)
plt.title('Gradient Descent Loss Curve (MSE)', fontsize=12)
plt.xlabel('Iterations', fontsize=10)
plt.ylabel('Loss (MSE)', fontsize=10)
plt.grid(alpha=0.3)
plt.show()