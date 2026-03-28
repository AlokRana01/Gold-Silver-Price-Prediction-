def create_features(df, target_col, metal_type):
    df = df.copy()

    df['Target'] = df[target_col]

    # Lag features
    df['lag_1'] = df['Target'].shift(1)
    df['lag_2'] = df['Target'].shift(2)
    df['rolling_mean_7'] = df['Target'].rolling(7).mean()

    # External feature
    df['usd_inr'] = df['USD_INR']

    # Date features
    df['day'] = df['Date'].dt.day
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year

    # Metal type
    df['metal_type'] = metal_type

    df = df.dropna()

    feature_cols = [
        'lag_1', 'lag_2', 'rolling_mean_7',
        'usd_inr', 'day', 'month', 'year',
        'metal_type'
    ]

    X = df[feature_cols]
    y = df['Target']

    return X, y