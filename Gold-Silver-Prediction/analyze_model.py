import joblib
from sklearn.ensemble import RandomForestRegressor

# Load the existing model
model_path = "models/metal_model.pkl"
try:
    model = joblib.load(model_path)
    
    # Calculate some stats
    n_estimators = len(model.estimators_)
    depths = [tree.get_depth() for tree in model.estimators_]
    leaves = [tree.get_n_leaves() for tree in model.estimators_]
    
    print(f"Model: {model_path}")
    print(f"Number of estimators: {n_estimators}")
    print(f"Average depth: {sum(depths) / n_estimators:.2f}")
    print(f"Max depth: {max(depths)}")
    print(f"Min depth: {min(depths)}")
    print(f"Average leaves: {sum(leaves) / n_estimators:.2f}")
    print(f"Max leaves: {max(leaves)}")
    print(f"Min leaves: {min(leaves)}")
except Exception as e:
    print(f"Error: {e}")
