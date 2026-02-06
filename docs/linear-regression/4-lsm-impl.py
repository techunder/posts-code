import numpy as np
import pandas as pd

# read the CSV file
df = pd.read_csv('lifespan_data_full.csv')

# show the first 5 rows of the dataframe
print(df.head())

# turn df into numpy.ndarray
X = df[['parent_lifespan', 'gender', 'exercise_hours', 'smoking', 'diet_health', 'sleep_hours', 'stress_level']].values
y = df['actual_lifespan'].values

print(X[:5])
print(y[:5])
