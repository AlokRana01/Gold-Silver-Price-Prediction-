import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

from src.preprocess import load_data
from src.features import create_features
from src.predict import load_model, forecast_next_days

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Gold & Silver Dashboard", layout="wide")
st.markdown("""
    <style>
    /* ── Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;700&family=Inter:wght@300;400;500;600&family=Rajdhani:wght@400;500;600;700&display=swap');

    /* ── Global body ── */
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', sans-serif !important;
    }

    /* ── Main title ── */
    h1 {
        font-family: 'Cinzel', serif !important;
        font-weight: 700 !important;
        letter-spacing: 1.5px !important;
    }

    /* ── Subheaders ── */
    h2, h3 {
        font-family: 'Cinzel', serif !important;
        font-weight: 600 !important;
        letter-spacing: 1px !important;
    }

    /* ── Tab labels ── */
    button[data-baseweb="tab"] {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        letter-spacing: 0.5px !important;
    }

    /* ── Sidebar labels & text ── */
    section[data-testid="stSidebar"] * {
        font-family: 'Inter', sans-serif !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        font-family: 'Cinzel', serif !important;
    }

    /* ── Metric values (prices, numbers) ── */
    [data-testid="stMetricValue"] {
        font-family: 'Rajdhani', sans-serif !important;
        font-weight: 700 !important;
        font-size: 2rem !important;
        letter-spacing: 0.5px !important;
    }
    [data-testid="stMetricLabel"] {
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        letter-spacing: 0.3px !important;
    }
    [data-testid="stMetricDelta"] {
        font-family: 'Rajdhani', sans-serif !important;
        font-weight: 600 !important;
    }

    /* ── Dataframe / table ── */
    .stDataFrame, .stDataFrame * {
        font-family: 'Rajdhani', sans-serif !important;
        font-size: 14px !important;
    }

    /* ── Caption / small text ── */
    .stCaption, small {
        font-family: 'Inter', sans-serif !important;
        font-style: italic;
        letter-spacing: 0.2px;
    }

    /* ── Selectbox & slider labels ── */
    label, .stSelectbox label, .stSlider label {
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
    }

    /* ── Buttons ── */
    div.stButton > button {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
        color: black !important;
        background-color: #FFD700 !important;
        border-radius: 8px;
        padding: 8px 16px;
    }
    div.stDownloadButton > button {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
        color: black !important;
        background-color: #00CFFF !important;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("💰 Gold & Silver Price Prediction Dashboard")
st.caption("AI-powered forecasting with economic indicators")

# ---------------- LOAD DATA ----------------
df = load_data("data/final_data.csv")

# ⚠️ Limit data for smooth animation
df = df.tail(200)

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Settings")

option = st.sidebar.selectbox("Select Metal", ["Gold 24K", "Gold 22K", "Silver"])
forecast_days = st.sidebar.slider("Forecast Days", 1, 14, 7)

# ---------------- TARGET + METAL TYPE ----------------
if option == "Gold 24K":
    target = "Gold_24K_1g"
    metal_type = 0
elif option == "Gold 22K":
    target = "Gold_22K_1g"
    metal_type = 1
else:
    target = "Silver_1g"
    metal_type = 2

# ---------------- LOAD MODEL ----------------
model = load_model("models/metal_model.pkl")

# ---------------- FEATURES ----------------
X, y = create_features(df, target, metal_type)
predictions = model.predict(X)

# ---------------- METRICS ----------------
latest = y.iloc[-1]
pred = predictions[-1]
change = pred - latest

# ---------------- KPI ----------------
st.subheader("💰 Key Metrics")

col1, col2, col3 = st.columns(3)
col1.metric("Current Price", f"₹ {latest:.2f}")
col2.metric("Predicted Price", f"₹ {pred:.2f}")
col3.metric("Change", f"₹ {change:.2f}")

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔮 Forecast", "📈 Insights"])

# ================= OVERVIEW =================
with tab1:

    col4, col5 = st.columns(2)

    # 📈 Actual vs Predicted
    with col4:
        st.subheader("📈 Actual vs Predicted")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['Date'][-len(y):],
            y=y,
            name='Actual',
            line=dict(width=3)
        ))
        fig.add_trace(go.Scatter(
            x=df['Date'][-len(y):],
            y=predictions,
            name='Predicted',
            line=dict(dash='dash')
        ))

        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    # 🎬 Animated Trend
    with col5:
        st.subheader("🎬 Animated Price Trend")

        fig2 = go.Figure(
            data=[go.Scatter(x=[df['Date'].iloc[0]], y=[df[target].iloc[0]])],
            layout=go.Layout(
                xaxis=dict(range=[df['Date'].min(), df['Date'].max()]),
                yaxis=dict(range=[df[target].min(), df[target].max()]),
                updatemenus=[{
                    "type": "buttons",
                    "bgcolor": "#FFD700",
                    "font": {"color": "black"},
                    "buttons": [{
                        "label": "▶ Play",
                        "method": "animate",
                        "args": [None, {
                            "frame": {"duration": 20, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 100}
                        }]
                    }]
                }]
            ),
            frames=[
                go.Frame(
                    data=[go.Scatter(
                        x=df['Date'][:k],
                        y=df[target][:k],
                        line=dict(width=3)
                    )]
                )
                for k in range(1, len(df))
            ]
        )

        st.plotly_chart(fig2, use_container_width=True)

# ================= FORECAST =================
with tab2:

    st.subheader(f"🔮 {forecast_days}-Day Forecast")

    last_row = X.iloc[-1].values
    future_preds = forecast_next_days(model, last_row, forecast_days)

    future_dates = pd.date_range(
        start=df['Date'].iloc[-1],
        periods=forecast_days + 1
    )[1:]

    # Build a rich, readable forecast DataFrame
    daily_changes = [future_preds[0] - latest] + [
        future_preds[i] - future_preds[i - 1] for i in range(1, len(future_preds))
    ]
    forecast_df = pd.DataFrame({
        "Day":            [f"Day {i+1}" for i in range(len(future_preds))],
        "Date":           future_dates.strftime("%d-%b-%Y"),
        "Metal":          option,
        "Forecast Price (₹)": [round(p, 2) for p in future_preds],
        "Daily Change (₹)":   [round(c, 2) for c in daily_changes],
        "Trend":          ["▲ Up" if c >= 0 else "▼ Down" for c in daily_changes],
    })

    # 🎬 Animated Forecast
    fig3 = go.Figure(
        data=[go.Scatter(x=[future_dates[0]], y=[future_preds[0]])],
        layout=go.Layout(
            xaxis=dict(range=[future_dates.min(), future_dates.max()]),
            yaxis=dict(range=[min(future_preds), max(future_preds)]),
            updatemenus=[{
                "type": "buttons",
                "bgcolor": "#FFD700",
                "font": {"color": "black"},
                "buttons": [{
                    "label": "▶ Play",
                    "method": "animate",
                    "args": [None, {
                        "frame": {"duration": 300, "redraw": True},
                        "fromcurrent": True,
                        "transition": {"duration": 200}
                    }]
                }]
            }]
        ),
        frames=[
            go.Frame(
                data=[go.Scatter(
                    x=future_dates[:k],
                    y=future_preds[:k],
                    mode='lines+markers',
                    line=dict(width=3)
                )]
            )
            for k in range(1, len(future_preds))
        ]
    )

    st.plotly_chart(fig3, use_container_width=True)
    st.dataframe(forecast_df)

    st.download_button(
        "📥 Download Forecast",
        forecast_df.to_csv(index=False),
        file_name="forecast.csv"
    )

# ================= INSIGHTS =================
with tab3:

    # ── 1. USD-INR Trend ──────────────────────────────────────
    st.subheader("🌍 USD-INR Exchange Rate Trend")

    col_c, col_d = st.columns([2, 1])
    with col_c:
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=df['Date'], y=df['USD_INR'],
            name='USD-INR',
            line=dict(color='#00CFFF', width=2.5),
            fill='tozeroy', fillcolor='rgba(0,207,255,0.08)'
        ))
        fig4.update_layout(template="plotly_dark", height=320)
        st.plotly_chart(fig4, use_container_width=True)

    with col_d:
        usd_latest = df['USD_INR'].iloc[-1]
        usd_min    = df['USD_INR'].min()
        usd_max    = df['USD_INR'].max()
        usd_mean   = df['USD_INR'].mean()
        st.markdown("#### 💱 Exchange Stats")
        st.metric("Latest Rate",  f"₹ {usd_latest:.2f}")
        st.metric("Min Rate",     f"₹ {usd_min:.2f}")
        st.metric("Max Rate",     f"₹ {usd_max:.2f}")
        st.metric("Average Rate", f"₹ {usd_mean:.2f}")

    st.divider()

    # ── 2. Statistical Summary Table ──────────────────────────
    st.subheader("📋 Price Statistics Summary")

    stats = df[target].describe().rename({
        'count': 'Count', 'mean': 'Mean (₹)', 'std': 'Std Dev (₹)',
        'min': 'Min (₹)', '25%': '25th Pct (₹)', '50%': 'Median (₹)',
        '75%': '75th Pct (₹)', 'max': 'Max (₹)'
    })
    st.dataframe(
        stats.to_frame(name=option).T.style.format("{:.2f}"),
        use_container_width=True
    )