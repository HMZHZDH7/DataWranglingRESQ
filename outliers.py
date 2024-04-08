# Import necessary libraries
import numpy as np
from sklearn.ensemble import IsolationForest
import pandas as pd

# Load the data
data = pd.read_csv('Data/dataREanonymized_long.csv')

# Filter data for the target site and variable
target_site = "Vitality"
target_variable = 'discharge_mrs'

filtered_data = data[(data['site_id'] == target_site) & (data['variable'] == target_variable)]

# Extract the feature (value) for anomaly detection
X_train = filtered_data['Value'].dropna().to_frame()  # Convert to DataFrame

# Create an Isolation Forest model
clf = IsolationForest(contamination=0.1, random_state=42)

# Fit the model to your data
clf.fit(X_train)

# Predict outliers/anomalies
y_pred = clf.predict(X_train)

# Get the indices of outliers
outlier_indices = np.where(y_pred == -1)[0]

# Print the indices of outliers
print("Indices of outliers:", outlier_indices)

# Convert 'Value' column to numeric
X_train['Value'] = pd.to_numeric(X_train['Value'], errors='coerce')

# Drop rows with missing or non-numeric values
X_train = X_train.dropna(subset=['Value'])

median_before = X_train['Value'].median()
print("Median before outlier removal:", median_before)

# Remove outliers
X_train_no_outliers = X_train[y_pred == 1]

# Calculate median after outlier removal
median_after = X_train_no_outliers['Value'].median()
print("Median after outlier removal:", median_after)
