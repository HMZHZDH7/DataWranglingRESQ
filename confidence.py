import numpy as np
from scipy import stats

group1 = np.array([85, 78, 90, 92, 180])
group2 = np.array([75, 5, 7, 8, 82])

# Calculate means and standard deviations
mean1, mean2 = np.mean(group1), np.mean(group2)
std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
n1, n2 = len(group1), len(group2)

# Calculate pooled standard deviation
pooled_std = np.sqrt(((n1-1)*(std1**2) + (n2-1)*(std2**2)) / (n1 + n2 - 2))

# Calculate standard error of the difference
standard_error_diff = pooled_std * np.sqrt((1/n1) + (1/n2))

# Define confidence level (e.g., 95%)
confidence_level = 0.95

# Calculate the margin of error
margin_of_error = stats.t.ppf((1 + confidence_level) / 2, n1 + n2 - 2) * standard_error_diff

# Calculate the confidence interval
lower_bound = (mean1 - mean2) - margin_of_error
upper_bound = (mean1 - mean2) + margin_of_error

print(mean1, mean2, mean1, mean2)
print((standard_error_diff))
print("Confidence Interval for the Difference of Means:", (lower_bound, upper_bound))

cohens_d = (mean1 - mean2) / pooled_std

print("Cohen's d:", cohens_d)

t_statistic, p_value = stats.ttest_ind(group1, group2)

print("t-statistic:", t_statistic)
print("p-value:", p_value)