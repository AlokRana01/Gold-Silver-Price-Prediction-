# ===============================
# IMPORTS
# ===============================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
import plotly.graph_objects as go

from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ===============================
# CONFIG
# ===============================
st.set_page_config(page_title="Metal Intelligence", layout="wide")

# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/final_data.csv")
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date']).sort_values("Date")
    return df

df = load_data()

if df.empty or len(df) < 40:
    st.error("Not enough data")
    st.stop()

# ===============================
# LOAD MODEL
# ===============================
model = joblib.load("models/model.pkl")

# ===============================
# FEATURE ENGINEERING
# ===============================
def create_features(df):
    df = df.copy()

    df["Lag_1"] = df["Gold_24K_1g"].shift(1)
    df["Lag_2"] = df["Gold_24K_1g"].shift(2)
    df["Lag_3"] = df["Gold_24K_1g"].shift(3)

    df["MA_7"] = df["Gold_24K_1g"].rolling(7).mean()
    df["MA_30"] = df["Gold_24K_1g"].rolling(30).mean()

    df["DayOfWeek"] = df["Date"].dt.dayofweek

    if "Silver_1g" in df.columns:
        df["Silver_Change"] = df["Silver_1g"].pct_change().fillna(0)

    if "USD_INR" in df.columns:
        df["USD_Change"] = df["USD_INR"].pct_change().fillna(0)

    return df

# ===============================
# HEADER
# ===============================
st.title("📊 Metal Intelligence Dashboard")

# ===============================
# MODEL PERFORMANCE
# ===============================
st.subheader("📊 Model Performance")

df_feat = create_features(df).dropna()

required = list(model.feature_names_in_)

for col in required:
    if col not in df_feat.columns:
        df_feat[col] = 0

X = df_feat[required]
y = df_feat["Gold_24K_1g"]

preds = model.predict(X)

mae = mean_absolute_error(y, preds)
rmse = np.sqrt(mean_squared_error(y, preds))
mape = np.mean(np.abs((y - preds) / y)) * 100
r2 = r2_score(y, preds)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Accuracy", f"{100-mape:.2f}%")
c2.metric("MAPE", f"{mape:.2f}%")
c3.metric("MAE", f"{mae:.2f}")
c4.metric("RMSE", f"{rmse:.2f}")
c5.metric("R²", f"{r2:.2f}")

# ===============================
# DOWNLOAD REPORT
# ===============================
report_df = pd.DataFrame({
    "Date": df_feat["Date"],
    "Actual": y,
    "Predicted": preds
})

st.download_button("📄 Download Report", report_df.to_csv(index=False), "report.csv")

# ===============================
# 🚀 FORECAST (TODAY → NEXT 7 DAYS)
# ===============================
def forecast(base_col, days=7):

    temp = df[base_col].tolist()
    preds = []
    future_dates = []

    last_known_date = df['Date'].iloc[-1]
    today = datetime.now().date()

    gap_days = (today - last_known_date.date()).days
    total_days = gap_days + days

    for i in range(total_days):

        lag1 = temp[-1]
        lag2 = temp[-2]
        lag3 = temp[-3]

        ma7 = np.mean(temp[-7:])
        ma30 = np.mean(temp[-30:])

        future_date = last_known_date + timedelta(days=i+1)
        dow = future_date.weekday()

        inp = pd.DataFrame([{
            "Lag_1": lag1,
            "Lag_2": lag2,
            "Lag_3": lag3,
            "MA_7": ma7,
            "MA_30": ma30,
            "DayOfWeek": dow,
            "Silver_Change": 0,
            "USD_Change": 0
        }])

        for col in model.feature_names_in_:
            if col not in inp.columns:
                inp[col] = 0

        inp = inp[model.feature_names_in_]

        pred = model.predict(inp)[0]
        temp.append(pred)

        if future_date.date() >= today:
            preds.append(pred)
            future_dates.append(future_date.date())

    return preds[:days], future_dates[:days]

# ===============================
# COMMON UI SECTION
# ===============================
def show_section(metal):

    st.subheader(metal)

    weight = st.selectbox("Select Weight", ["1g", "10g", "100g", "1kg"], key=metal)
    col_name = f"{metal}_{weight}"

    latest = df.iloc[-1]
    previous = df.iloc[-2]

    today_price = latest[col_name]
    yesterday = previous[col_name]

    change = today_price - yesterday

    a, b, c, d = st.columns(4)
    a.metric("Today", f"₹ {today_price:.2f}", f"{change:.2f}")
    b.metric("Yesterday", f"₹ {yesterday:.2f}")
    c.metric("Max", f"₹ {df[col_name].max():.2f}")
    d.metric("Min", f"₹ {df[col_name].min():.2f}")

    # ===============================
    # FORECAST
    # ===============================
    base_preds, future_dates = forecast("Gold_24K_1g")

    if metal == "Gold_22K":
        preds = [p*(22/24) for p in base_preds]
    elif metal == "Silver":
        ratio = df["Gold_24K_1g"].iloc[-1] / df["Silver_1g"].iloc[-1]
        preds = [p/ratio for p in base_preds]
    else:
        preds = base_preds

    multiplier = {"1g":1, "10g":10, "100g":100, "1kg":1000}[weight]
    preds = [p*multiplier for p in preds]

    # ===============================
    # 📊 FORECAST TABLE
    # ===============================
    st.markdown("### 📊 7-Day Forecast")

    rows = []
    for i in range(len(preds)):
        prev_val = today_price if i == 0 else preds[i-1]
        curr = preds[i]
        diff = curr - prev_val

        arrow = "▲" if diff > 0 else "▼"

        rows.append({
            "Date": future_dates[i],
            "Predicted Price": f"₹ {curr:.2f}",
            "Change": f"{arrow} {abs(diff):.2f}"
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # ===============================
    # GRAPH
    # ===============================
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df[col_name], name="Actual"))
    fig.add_trace(go.Scatter(x=future_dates, y=preds, name="Forecast"))

    st.plotly_chart(fig, use_container_width=True)

# ===============================
# TABS
# ===============================
tab1, tab2, tab3, tab4 = st.tabs(["Gold 24K", "Gold 22K", "Silver", "USD"])

with tab1:
    show_section("Gold_24K")

with tab2:
    show_section("Gold_22K")

with tab3:
    show_section("Silver")

with tab4:
    st.subheader("USD-INR")

    usd = yf.download("USDINR=X", period="1y")

    if isinstance(usd.columns, pd.MultiIndex):
        usd.columns = usd.columns.get_level_values(0)

    usd.reset_index(inplace=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=usd['Date'], y=usd['Close'], name="USD-INR"))

    st.plotly_chart(fig, use_container_width=True)

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.caption("© 2026 Metal Intelligence")