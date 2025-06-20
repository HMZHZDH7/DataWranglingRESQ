import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import networkx as nx
matplotlib.use('TkAgg')  # or 'Qt5Agg' if you have PyQt5 installed
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from dtreeviz import model as dtreeviz_model


# 1. Load trained model
model_path = 'Data/XGboostCat20250620_133457.pkl'  # Replace with your actual file if needed
xg = joblib.load(model_path)

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

X_full = X_full.drop(columns=['subject_id'], errors='ignore')
X_patient = X_patient.drop(columns=['subject_id'], errors='ignore')

# Predict the class and probability
y_pred = xg.predict(X_patient)[0]
y_proba = xg.predict_proba(X_patient)[0][1]  # probability of mRS ≤ 2

print(f"Prediction for subject {subject_id}: mRS ≤ 2 = {bool(y_pred)}, probability = {y_proba:.2f}")

surrogate = DecisionTreeClassifier(max_depth=5, random_state=42)
surrogate.fit(X_full, xg.predict(X_full))

# Visualize it
plt.figure(figsize=(20, 10))
plot_tree(surrogate, feature_names=X_full.columns, class_names=['mRS > 2', 'mRS ≤ 2'], filled=True, rounded=True)
plt.title("Surrogate Tree Explaining XGBoost Predictions")
plt.show()

# Show decision path for selected patient
node_indicator = surrogate.decision_path(X_patient)
leaf_id = surrogate.apply(X_patient)

feature = surrogate.tree_.feature
threshold = surrogate.tree_.threshold

print("\nDecision path for the selected patient:")
for node_id in node_indicator.indices:
    if leaf_id[0] == node_id:
        continue  # Skip leaf node
    feature_name = X_patient.columns[feature[node_id]]
    threshold_value = threshold[node_id]
    patient_value = X_patient.iloc[0, feature[node_id]]
    decision = "<=" if patient_value <= threshold_value else ">"
    print(f"Node {node_id}: ({feature_name} = {patient_value}) {decision} {threshold_value}")


