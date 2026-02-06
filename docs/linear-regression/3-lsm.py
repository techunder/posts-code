import numpy as np
import pandas as pd

true_b = 10  # Base lifespan value
true_w = np.array([
    0.4,    # Parent average lifespan weight
    6,      # Gender weight
    0.8,    # Weekly exercise duration weight
    -15,    # Smoking weight (negative, largest absolute value)
    9,      # Diet health weight
    1.3,    # Daily sleep duration weight
    7       # Stress/mental health weight
])

# read the CSV file
df = pd.read_csv('lifespan_data_full.csv')

# show the first 5 rows of the dataframe
print(df.head())

# turn df into numpy.ndarray
X = df[['parent_lifespan', 'gender', 'exercise_hours', 'smoking', 'diet_health', 'sleep_hours', 'stress_level']].values
y = df['actual_lifespan'].values

## 添加截距项（全1列）
X_with_bias = np.column_stack([np.ones(len(X)), X])

## 计算(X^T X)的逆矩阵
X_T = X_with_bias.T
X_T_X = X_T @ X_with_bias

## 检查矩阵是否可逆（科普用）
if np.linalg.matrix_rank(X_T_X) != X_T_X.shape[0]:
    raise ValueError("X^TX矩阵不可逆，无法计算解析解！")

# 计算最优参数
theta = np.linalg.inv(X_T_X) @ X_T @ y

w = theta[:-1]  # 权重项
b = theta[-1]  # 截距项

print("\n===== 解析解求解结果 =====")
print(f"解析解截距b：{b:.4f}（真实值：{true_b}）")
print(f"解析解权重w：{np.round(w, 4)}（真实值：{true_w}）")
