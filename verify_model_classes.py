
import catboost
import numpy as np

try:
    print("Loading model...")
    model = catboost.CatBoostClassifier()
    model.load_model('CatBoost_Best_Model.cbm')
    
    print(f"Classes: {model.classes_}")
    
except Exception as e:
    print(f"Error: {e}")
