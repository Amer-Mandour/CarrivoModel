
import pickle
import sys
import os

print(f"Python version: {sys.version}")

try:
    print("Loading Best_Model.pkl...")
    with open('Best_Model.pkl', 'rb') as f:
        model_pkl = pickle.load(f)
    print(f"Type of Best_Model.pkl: {type(model_pkl)}")
    if hasattr(model_pkl, 'classes_'):
        print(f"Classes in pickle model: {model_pkl.classes_}")
    if hasattr(model_pkl, 'feature_names_in_'):
        print(f"Feature names in pickle model: {model_pkl.feature_names_in_}")
except Exception as e:
    print(f"Error loading pickle: {e}")

try:
    import catboost
    print("\nLoading CatBoost_Best_Model.cbm...")
    model_cb = catboost.CatBoostClassifier()
    model_cb.load_model('CatBoost_Best_Model.cbm')
    print(f"Type of CatBoost model: {type(model_cb)}")
    print(f"Classes in CatBoost model: {model_cb.classes_}")
    print(f"Feature names in CatBoost model: {model_cb.feature_names_}")
except ImportError:
    print("\nCatboost not installed, skipping .cbm check")
except Exception as e:
    print(f"Error loading CatBoost: {e}")
