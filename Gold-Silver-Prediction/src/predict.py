import joblib

def load_model(path):
    return joblib.load(path)

def forecast_next_days(model, last_row, days=7):
    preds = []
    current = last_row.copy()

    for _ in range(days):
        pred = model.predict([current])[0]
        preds.append(pred)

        # update lag values
        current[1] = current[0]
        current[0] = pred

    return preds