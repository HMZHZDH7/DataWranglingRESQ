import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg' if you have PyQt5 installed
import seaborn as sns

# 1. Load trained model
model_path = 'Data/XGboostCat20250620_133457.pkl'  # Replace with your actual file if needed
model = joblib.load(model_path)

# 2. Load and filter long-format data
df_long = pd.read_csv("Data/dataREanonymized_long.csv")

valid_subjects = df_long[
    (df_long['variable'] == 'discharge_mrs') &
    (df_long['Value'].notna())
]['subject_id'].unique()

df_long_filtered = df_long[df_long['subject_id'].isin(valid_subjects)]

# 3. Pivot to wide format
df_wide = df_long_filtered.pivot(index='subject_id', columns='variable', values='Value').reset_index()

# 4. Filter: ischemic stroke + thrombolysis = 1
df_wide = df_wide[df_wide['stroke_type'] == 'ischemic']
df_wide = df_wide[df_wide['thrombolysis'] == '1']

# 5. Convert target
target = pd.to_numeric(df_wide['discharge_mrs'], errors='coerce')
y = (target <= 2).astype(int)

# 6. Prepare features
X_full = df_wide.drop(columns=['discharge_mrs', 'three_m_mrs', 'stroke_type', 'door_to_groin', 'thrombolysis',
                          'bleeding_source', 'bleeding_volume_value', 'department_type', 'dysphagia_screening_type',
                          'gender', 'hospitalized_in', 'hunt_hess_score', 'ich_score', 'imaging_type', 'no_thrombolysis_reason',
                          'stroke_mimics_diagnosis'])

X_full = X_full.dropna(axis=1, how='all')


# Convert to numeric and fill missing
X_full = X_full.apply(pd.to_numeric, errors='coerce')

X_full = X_full.fillna(X_full.median(numeric_only=True))

print(X_full)

# 7. Randomly select a patient
random_index = np.random.randint(0, len(X_full))
X_patient = X_full.iloc[[random_index]]
subject_id = df_wide.iloc[random_index]['subject_id']

X_patient = X_patient.drop(columns=['subject_id'], errors='ignore')

# 8. Predict
prediction_proba = model.predict_proba(X_patient)[0][1]
prediction_class = model.predict(X_patient)[0]

print(f"Explaining subject ID: {subject_id}")
print(f"Predicted probability of poor outcome (mRS > 2): {prediction_proba:.2f}")
print(f"Predicted class: {prediction_class}")

# 9. SHAP explanation
explainer = shap.Explainer(model)
shap_values = explainer(X_patient)

# Visualize
shap.initjs()
shap.force_plot(shap_values.base_values[0], shap_values.values[0], X_patient.iloc[0], matplotlib=True)
plt.show()

shap.plots.waterfall(shap_values[0])

# If shap_values is an Explanation object (e.g., from shap.Explainer)
shap_values_patient = shap_values[0]  # for a single patient

# Create a DataFrame using shap_values_patient.values instead of the object directly
shap_df = pd.DataFrame({
    'feature': X_patient.columns,
    'shap_value': shap_values_patient.values  # <-- fix here
})

# Now you can take the absolute value
shap_df['abs_val'] = shap_df['shap_value'].abs()

# Sort and plot top 10
shap_df_sorted = shap_df.sort_values(by='abs_val', ascending=False).head(10)

plt.figure(figsize=(10, 6))
colors = shap_df_sorted['shap_value'].apply(lambda x: 'red' if x > 0 else 'blue')
plt.barh(shap_df_sorted['feature'], shap_df_sorted['shap_value'], color=colors)
plt.xlabel("Effect on predicted risk")
plt.title("Top 10 factors influencing prediction")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# Prepare SHAP data
shap_values_patient = shap_values[0]
shap_df = pd.DataFrame({
    'feature': X_patient.columns,
    'shap_value': shap_values_patient.values,
    'feature_value': X_patient.iloc[0].values
})
shap_df['abs_val'] = shap_df['shap_value'].abs()
shap_df_sorted = shap_df.sort_values(by='abs_val', ascending=False).head(10)

# Add readable feature labels with values
shap_df_sorted['label'] = shap_df_sorted.apply(
    lambda row: f"{row['feature']}: {row['feature_value']}", axis=1
)

# Determine bar colors manually (based on SHAP value sign)
colors = ['indianred' if val > 0 else 'steelblue' for val in shap_df_sorted['shap_value']]

# Plot
plt.figure(figsize=(10, 6))
bars = plt.barh(
    y=shap_df_sorted['label'],
    width=shap_df_sorted['shap_value'],
    color=colors,
    edgecolor='black'
)

# Add text labels on bars
for bar, val in zip(bars, shap_df_sorted['shap_value']):
    x = bar.get_width()
    ha = 'left' if x > 0 else 'right'
    offset = 0.02 if x > 0 else -0.02
    plt.text(x + offset, bar.get_y() + bar.get_height()/2, f"{x:.2f}",
             va='center', ha=ha, fontsize=9)

plt.axvline(0, color='gray', linewidth=0.8)
plt.title("Top 10 Factors Influencing Prediction", fontsize=14)
plt.xlabel("Impact on Model Output (SHAP Value)", fontsize=12)
plt.ylabel("")
plt.tight_layout()
plt.gca().invert_yaxis()  # Top = most important
plt.show()