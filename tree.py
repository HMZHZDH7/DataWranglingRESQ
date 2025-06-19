from catboost import CatBoostClassifier, Pool
import pickle
import matplotlib.pyplot as plt
# Load model from pickle
with open("Data/AIS_thrombolysis_only_binary_model_full_cases.pkl", "rb") as f:
    model = pickle.load(f)

model.plot_tree(tree_idx=0)
plt.show()