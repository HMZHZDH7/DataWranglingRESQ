import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
import os
from catboost import CatBoostClassifier

# Load data
df_long = pd.read_csv('Data\XGboostCat20250620_130110.pkl')

# Filter for valid target
valid_subjects = df_long[
    (df_long['variable'] == 'discharge_mrs') & df_long['Value'].notna()
]['subject_id'].unique()
df_long_filtered = df_long[df_long['subject_id'].isin(valid_subjects)]

# Pivot to wide
df_wide = df_long_filtered.pivot(index='subject_id', columns='variable', values='Value').reset_index()

# Filter for ischemic stroke and thrombolysis
df_wide = df_wide[df_wide['stroke_type'] == 'ischemic']
df_wide = df_wide[df_wide['thrombolysis'] == '1']

# Target
target = pd.to_numeric(df_wide['discharge_mrs'], errors='coerce')
y = (target <= 2).astype(int)

# Features
X = df_wide.drop(columns=['discharge_mrs', 'three_m_mrs', 'stroke_type', 'door_to_groin', 'thrombolysis'])
X = X.apply(pd.to_numeric, errors='coerce')
X = X.fillna(X.median(numeric_only=True))

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Hyperparameter grid for CatBoost
param_grid = {
    'iterations': [100, 200, 300],
    'depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'l2_leaf_reg': [1, 3, 5, 7],
    'border_count': [32, 64, 128],
    'bagging_temperature': [0, 1, 5],
    'random_strength': [0.5, 1, 2]
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Initialize CatBoost (silent for cleaner output)
model = CatBoostClassifier(
    loss_function='Logloss',
    eval_metric='F1',
    auto_class_weights='Balanced',
    verbose=0,
    random_state=42
)

# Random search
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_grid,
    n_iter=50,
    scoring='f1',
    cv=cv,
    verbose=2,
    n_jobs=-1
)

# Fit
random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_

# Evaluate
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))

# Feature importance
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': best_model.get_feature_importance()
}).sort_values(by='importance', ascending=False)

print(importance_df.head(10))

print("Best hyperparameters:")
for param, value in random_search.best_params_.items():
    print(f"{param}: {value}")

# Save
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = os.path.join("Data", f"CatBoostModel_{timestamp}.pkl")
joblib.dump(best_model, output_path)
print(f"Model saved to: {output_path}")
