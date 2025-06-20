import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg' if you have PyQt5 installed
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import joblib
from datetime import datetime
import os


df_long = pd.read_csv('Data/dataREanonymized_long.csv')

# Step 1: Filter out rows where variable == 'three_m_mrs' and Value is missing
# First, identify subjects that *do* have a valid `three_m_mrs` value
valid_subjects = df_long[
    (df_long['variable'] == 'discharge_mrs') &
    (df_long['Value'].notna())
]['subject_id'].unique()

# Step 2: Keep only rows for those valid subjects
df_long_filtered = df_long[df_long['subject_id'].isin(valid_subjects)]

# 2. Pivot to wide format
df_wide = df_long_filtered.pivot(index='subject_id', columns='variable', values='Value').reset_index()

# Keep only rows where stroke_type is "ischemic"
df_wide = df_wide[df_wide['stroke_type'] == 'ischemic']
df_wide = df_wide[df_wide['thrombolysis'] == '1']

print(df_wide)

# 3. Target variable extraction
# If `three_m_mrs` is stored as a row in the long format, extract it:
#target = df_wide['discharge_mrs'].to_numeric(df_wide['discharge_mrs'], errors='coerce')
target = pd.to_numeric(df_wide['discharge_mrs'], errors='coerce')

X = df_wide.drop(columns=['discharge_mrs', 'three_m_mrs', 'stroke_type', 'door_to_groin', 'thrombolysis', 'subject_id',
                          'bleeding_source', 'bleeding_volume_value', 'department_type', 'dysphagia_screening_type',
                          'gender', 'hospitalized_in', 'hunt_hess_score', 'ich_score', 'imaging_type', 'no_thrombolysis_reason',
                          'stroke_mimics_diagnosis'])

X = X.dropna(axis=1, how='all')

# 4. Preprocessing
# Convert all data to numeric if possible
X = X.apply(pd.to_numeric, errors='coerce')

# Handle missing values (simple strategy here: fill with mean)
X = X.fillna(X.median(numeric_only=True))

y = (target <= 2).astype(int)
print(y.value_counts())

# 5. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 1, 5],
    'min_child_weight': [1, 3, 5],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 1.5, 2]
}


cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)




# 6. Train XGBoost
model = XGBClassifier(objective='binary:logistic', eval_metric='mlogloss', class_weight='balanced')

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_grid,
    n_iter=50,
    scoring='f1',
    cv=cv,
    verbose=2,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_

# 7. Evaluate
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))

importance_dict = best_model.get_booster().get_score(importance_type='weight')  # or 'gain', 'cover'

# Convert to DataFrame and sort
importance_df = pd.DataFrame({
    'feature': list(importance_dict.keys()),
    'importance': list(importance_dict.values())
}).sort_values(by='importance', ascending=False)

print(importance_df.head(10))

print("Best hyperparameters:")
for param, value in random_search.best_params_.items():
    print(f"{param}: {value}")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Define output path
output_path = os.path.join("Data", f"XGboostCat{timestamp}.pkl")

# Save the model
joblib.dump(best_model, output_path)

print(f"Model saved to: {output_path}")