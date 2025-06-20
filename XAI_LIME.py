import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg' if you have PyQt5 installed
import seaborn as sns
from lime.lime_tabular import LimeTabularExplainer
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

# Convert to numeric and fill missing
X_full = X_full.apply(pd.to_numeric, errors='coerce')
X_full = X_full.fillna(X_full.median(numeric_only=True))

# 7. Randomly select a patient
random_index = np.random.randint(0, len(X_full))



X_patient = X_full.iloc[[random_index]]
subject_id = df_wide.iloc[random_index]['subject_id']

X_full = X_full.drop(columns=['subject_id'], errors='ignore')
X_patient = X_patient.drop(columns=['subject_id'], errors='ignore')

X_full = X_full.apply(pd.to_numeric, errors='coerce')
X_patient = X_patient.apply(pd.to_numeric, errors='coerce')

X_full = X_full.fillna(X_full.median(numeric_only=True))
X_patient = X_patient.fillna(X_patient.median(numeric_only=True))

# Show columns with NaNs and their counts
nan_counts = X_full.isna().sum()
nan_columns = nan_counts[nan_counts > 0]

print("Columns with NaNs:")
print(nan_columns.sort_values(ascending=False))

assert not X_patient.isnull().values.any(), "X_patient still has NaNs!"

# Predict the outcome
y_pred_proba = model.predict_proba(X_patient)[0][1]
y_pred_class = model.predict(X_patient)[0]

print(f"Prediction for subject_id {subject_id}:")
print(f"  Probability of good outcome (mRS ≤ 2): {y_pred_proba:.2f}")
print(f"  Predicted class: {y_pred_class}")

# Workaround: Add small noise to zero-variance columns (only for LIME)
X_lime = X_full.copy()
stds = X_lime.std(numeric_only=True)
zero_var_cols = stds[stds == 0].index

# Add tiny noise ONLY to zero-variance columns
for col in zero_var_cols:
    X_lime[col] += np.random.normal(0, 1e-6, size=len(X_lime))

X_patient_lime = X_lime.iloc[[random_index]]

print(X_full)
print(X_patient)
explainer = LimeTabularExplainer(
    training_data=np.array(X_full),
    feature_names=X_full.columns.tolist(),
    class_names=['Poor Outcome', 'Good Outcome'],
    mode='classification',
    discretize_continuous=True  # << turn OFF discretization entirely
)

exp = explainer.explain_instance(
    data_row=X_patient.iloc[0],
    predict_fn=model.predict_proba,
    num_features=10
)

# or
exp.save_to_file("lime_explanation.html")