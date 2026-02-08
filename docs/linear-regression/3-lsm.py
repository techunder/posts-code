# Using Least Squares Method (LSM) to Solve Linear Regression

import numpy as np
import pandas as pd

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

# append bias (1 column of ones)
X_with_bias = np.column_stack([X, np.ones(len(X))])

# calculate X^T
X_T = X_with_bias.T

# calculate X^T X
X_T_X = X_T @ X_with_bias

# check if X^T X is invertible
if np.linalg.matrix_rank(X_T_X) != X_T_X.shape[0]:
    raise ValueError("X^TX matrix is not invertible!")

# calculate (X^T X)^{-1}
theta = np.linalg.inv(X_T_X) @ X_T @ y

# extract the weights and intercept
w = theta[:-1]  # weights
b = theta[-1]  # intercept (bias)

# print the weights and intercept
print("\n===== Linear Regression Solution =====")
print(f"Intercept (bias): {b:.4f}")
print(f"Weights: {np.round(w, 4)}")