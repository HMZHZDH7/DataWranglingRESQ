import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import seaborn as sns
from alibi.explainers import AnchorTabular
matplotlib.use('TkAgg')  # or 'Qt5Agg' if you have PyQt5 installed

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
                               'bleeding_source', 'bleeding_volume_value', 'department_type',
                               'dysphagia_screening_type',
                               'gender', 'hospitalized_in', 'hunt_hess_score', 'ich_score', 'imaging_type',
                               'no_thrombolysis_reason',
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

# 8. Make prediction
prediction = model.predict(X_patient)[0]
proba = model.predict_proba(X_patient)[0][1]  # Probability of mRS ≤ 2

print(
    f"Prediction for subject {subject_id}: {'Good outcome (mRS ≤ 2)' if prediction == 1 else 'Poor outcome (mRS ≥ 3)'}")
print(f"Probability of good outcome: {proba:.2f}")

feature_names = X_full.drop(columns=['subject_id'], errors='ignore').columns.tolist()


predict_fn = lambda x: model.predict(x)

# Fit explainer on training data
explainer = AnchorTabular(predict_fn, feature_names=feature_names)
explainer.fit(X_full.drop(columns=['subject_id'], errors='ignore').values, disc_perc=(25, 50, 75))

# 10. Explain the selected patient
explanation = explainer.explain(X_patient.values[0], threshold=0.99, beam_size=10)

print(f"Anchor explanation for subject {subject_id}:")
print('\n'.join(explanation.anchor))
print(f"Precision: {explanation.precision:.2f}")
print(f"Coverage: {explanation.coverage:.2f}")

# Basic anchor info
anchor_conditions = explanation.anchor
precision = explanation.precision
coverage = explanation.coverage

# Create a figure
fig, ax = plt.subplots(figsize=(10, 4))

# Plot each anchor condition as a separate bar
bars = ax.bar(anchor_conditions, [precision]*len(anchor_conditions), color=plt.cm.Blues(coverage))

# Annotate with precision and coverage
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'Prec: {precision:.2f}\nCov: {coverage:.2f}',
            ha='center', va='bottom', fontsize=10)

# Axis formatting
ax.set_ylim(0, 1.1)
ax.set_ylabel("Precision")
ax.set_title(f"Anchor Explanation for Subject {subject_id}")
ax.set_xticks(range(len(anchor_conditions)))
ax.set_xticklabels(anchor_conditions, rotation=45, ha='right')
plt.tight_layout()

plt.show()