from src.preprocess import load_data
from src.features import create_features
from src.train import train_model
import pandas as pd

df = load_data("data/final_data.csv")

X_all = []
y_all = []

# Gold 24K → 0
X, y = create_features(df, "Gold_24K_1g", 0)
X_all.append(X)
y_all.append(y)

# Gold 22K → 1
X, y = create_features(df, "Gold_22K_1g", 1)
X_all.append(X)
y_all.append(y)

# Silver → 2
X, y = create_features(df, "Silver_1g", 2)
X_all.append(X)
y_all.append(y)

X_final = pd.concat(X_all)
y_final = pd.concat(y_all)

train_model(X_final, y_final, "models/metal_model.pkl")

print("✅ Single model trained successfully!")