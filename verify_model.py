
import catboost
import numpy as np

try:
    print("Loading model...")
    model = catboost.CatBoostClassifier()
    model.load_model('CatBoost_Best_Model.cbm')
    
    print(f"Classes: {model.classes_}")
    print(f"Feature names: {model.feature_names_}")
    
    # Create dummy input (42 features)
    dummy_input = [3] * 42 # Neutral answers
    
    # Predict
    pred = model.predict(dummy_input)
    proba = model.predict_proba(dummy_input)
    
    print(f"Prediction: {pred}")
    print(f"Probabilities shape: {proba.shape}")
    print(f"Top 5 classes indices: {np.argsort(proba)[0][-5:][::-1]}")
    
except Exception as e:
    print(f"Error: {e}")
