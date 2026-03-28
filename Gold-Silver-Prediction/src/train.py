from sklearn.ensemble import RandomForestRegressor
import joblib

def train_model(X, y, path):
    model = RandomForestRegressor(
        n_estimators=50,
        max_depth=10,
        random_state=42
    )

    model.fit(X, y)

    joblib.dump(model, path, compress=3)

    return model