# Generate a Test Dataset for Linear Regression Model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# == 1. Generate sample size (1000 samples, balance between efficiency and representativeness)
n_samples = 1000

# == 2. Generate 7 independent variables (simulate according to authoritative ranges)
parent_lifespan = np.random.normal(75, 5, n_samples)  # Parent average lifespan: 75±5 years
gender = np.random.randint(0, 2, n_samples)           # Gender: 0=male, 1=female
exercise_hours = np.random.uniform(0, 10, n_samples)  # Weekly exercise duration: 0-10 hours
smoking = np.random.randint(0, 2, n_samples)          # Smoking: 0=non-smoker, 1=smoker
diet_health = np.random.randint(0, 2, n_samples)      # Diet health: 0=unhealthy, 1=healthy
sleep_hours = np.random.uniform(5, 9, n_samples)      # Daily sleep duration: 5-9 hours
stress_level = np.random.randint(0, 3, n_samples)     # Stress level: 0=high, 1=moderate, 2=low

# == 3. Set true weights (core parameters)
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

# == 4. Construct feature matrix X and dependent variable y (add random noise to simulate real-world differences)
X = np.column_stack([
    parent_lifespan, gender, exercise_hours, 
    smoking, diet_health, sleep_hours, stress_level
])

# == 5. Calculate actual lifespan: y = b + X·w + random noise (std=3, simulate individual differences)
y = true_b + np.dot(X, true_w)
#y = true_b + np.dot(X, true_w) + np.random.normal(0, 3, n_samples)

# == 6. Wrap into DataFrame and save
data = pd.DataFrame({
    'parent_lifespan': parent_lifespan,
    'gender': gender,
    'exercise_hours': exercise_hours,
    'smoking': smoking,
    'diet_health': diet_health,
    'sleep_hours': sleep_hours,
    'stress_level': stress_level,
    'actual_lifespan': y
})
data.to_csv('./lifespan_data_full.csv', index=False)

# == 7. View dataset basic information
print("===== Dataset Basic Information =====")
print(f"Dataset Shape: {data.shape}")
print("\nFirst 5 Rows of the Dataset:")
print(data.head())
print("\nDataset Descriptive Statistics:")
print(data.describe())