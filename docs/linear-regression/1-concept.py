# Generate Simulated Data Fitting WHO Standard

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['axes.unicode_minus'] = False  # fix minus display issue

# == 1. Generate Simulated Data Fitting WHO Standard
np.random.seed(42)  # fix random seed for reproducibility
n_people = 200      # 200 samples
max_exercise = 5    # WHO recommended upper limit for moderate intensity exercise: 5 hours/week (300 minutes)
mean_exercise = 2.5 # WHO base recommendation for moderate intensity exercise: 2.5 hours/week (150 minutes)

# Generate exercise duration data: normal distribution (mean 2.5, std 1), more realistic than uniform distribution
exercise_hours = np.random.normal(loc=mean_exercise, scale=1, size=n_people)
# Clip values to ensure they are within the valid range: 0 ≤ exercise_hours ≤ 5 hours
exercise_hours = np.clip(exercise_hours, 0, max_exercise)

# Generate lifespan data: base lifespan 70 years + decreasing marginal benefit of exercise + random noise
base_lifespan = 70  # base lifespan
# Exercise benefit: use square root function to implement "marginal benefit decreasing" (benefit decreases as exercise duration increases)
exercise_benefit = 8 * np.sqrt(exercise_hours / max_exercise)  # benefit of 5 hours of exercise is about 8 years
# Random noise: normal distribution (mean 0, std 1.5), to simulate individual variability
random_noise = np.random.normal(loc=0, scale=1.5, size=n_people)
lifespan = base_lifespan + exercise_benefit + random_noise

# == 2. Plot Scatter Plot with Trend Line
plt.figure(figsize=(10, 6))

# Plot scatter plot
plt.scatter(
    exercise_hours, lifespan, 
    color='steelblue', alpha=0.7, s=50,
    label='Personal Data'
)

# Fit linear trend line
z = np.polyfit(exercise_hours, lifespan, 1)
p = np.poly1d(z)
# Generate smooth x-axis data for trend line, to make it more visually appealing
x_smooth = np.linspace(0, max_exercise, 100)
plt.plot(
    x_smooth, p(x_smooth), 
    color='crimson', linewidth=2, 
    label='Trend Line'
)

# Annotate WHO recommended interval
plt.axvspan(2.5, 5, color='lightgreen', alpha=0.3, label='WHO Recommended Interval (2.5-5 hours/week)')
plt.axvline(x=2.5, color='forestgreen', linestyle='--', alpha=0.8, label='WHO Base Recommendation (2.5 hours/week)')

# == 3. Add Annotations
plt.title('The relationship between weekly exercise duration and lifespan (simulated data)', fontsize=14, pad=15)
plt.xlabel('Medium intensity exercise duration per week (hours)', fontsize=12)
plt.ylabel('Life span (years)', fontsize=12)
plt.xlim(-0.2, max_exercise + 0.2)  # x-axis range
plt.ylim(68, 82)  # y-axis range
plt.grid(True, alpha=0.3)
plt.legend(loc='lower right')

# Show plot
plt.tight_layout()
plt.show()

# Output core statistics
corr = np.corrcoef(exercise_hours, lifespan)[0, 1]
print(f'Correlation between weekly exercise duration and lifespan: {corr:.2f} (positive correlation, as expected)')
print(f'Average weekly exercise duration: {np.mean(exercise_hours):.1f} hours/week (close to WHO base recommendation of 2.5 hours)')
print(f'Average lifespan: {np.mean(lifespan):.1f} years')
