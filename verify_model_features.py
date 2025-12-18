
import catboost
import numpy as np

try:
    print("Loading model...")
    model = catboost.CatBoostClassifier()
    model.load_model('CatBoost_Best_Model.cbm')
    
    print(f"Feature count: {len(model.feature_names_)}")
    print(f"Feature names: {model.feature_names_}")
    
except Exception as e:
    print(f"Error: {e}")
