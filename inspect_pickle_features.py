
import pickle
import sys
import catboost

print(f"Python version: {sys.version}")

try:
    print("Loading Best_Model.pkl...")
    with open('Best_Model.pkl', 'rb') as f:
        model_pkl = pickle.load(f)
    
    print(f"Type of Best_Model.pkl: {type(model_pkl)}")
    
    if hasattr(model_pkl, 'feature_names_in_'):
        print(f"Feature names in pickle (len={len(model_pkl.feature_names_in_)}):")
        print(list(model_pkl.feature_names_in_))
    elif hasattr(model_pkl, 'get_feature_names_out'):
        print(f"Feature names (out): {model_pkl.get_feature_names_out()}")
    elif hasattr(model_pkl, 'feature_names_'):
        print(f"Feature names in pickle (len={len(model_pkl.feature_names_)}):")
        print(model_pkl.feature_names_)
    else:
        print("Could not find standard feature names attribute on pickle model.")
        print(f"Dir: {dir(model_pkl)}")

except Exception as e:
    print(f"Error loading pickle: {e}")
