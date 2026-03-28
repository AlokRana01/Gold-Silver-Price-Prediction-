# 💰 Gold & Silver Price Prediction Dashboard

A premium, AI-powered interactive dashboard designed to forecast **Gold (24K, 22K)** and **Silver prices**. Built with Python and Machine Learning, this tool leverages historical price data and global economic indicators (USD-INR) to provide actionable market insights.

---

## ✨ Key Features

*   **Multi-Asset Support**: Predict prices for Gold 24K, Gold 22K, and Silver (per 1 gram).
*   **AI Forecasting**: Smart 14-day future price prediction using lag-based feature engineering.
*   **Premium Visuals**: High-end UI with custom typography (**Cinzel**, **Inter**, **Rajdhani**) and **Plotly Dark Mode** charts.
*   **Dynamic Trends**: Animated price trend analysis and automated forecast visualizations.
*   **Market Insights**: Real-time USD-INR impact monitoring and statistical price summaries.
*   **Rich CSV Export**: Download forecasts in a formatted, readable CSV with trend indicators (▲/▼).

---

## 🏗️ Project Structure

```text
Gold-Silver-Prediction/
├── data/
│   └── final_data.csv        # Historical price and USD-INR dataset
├── models/
│   └── metal_model.pkl       # Trained ML model (Random Forest)
├── src/                      # Backend Logic
│   ├── preprocess.py         # Data loading and cleaning
│   ├── features.py           # Lag-based feature engineering
│   ├── train.py              # Model training script
│   └── predict.py            # Recursive forecasting logic
├── app.py                    # Main Streamlit Dashboard
├── main.py                   # Automated training entry point
├── requirements.txt          # Project dependencies
└── README.md
```

---

## ⚙️ Tech Stack

*   **UI/UX**: [Streamlit](https://streamlit.io/), [Plotly](https://plotly.com/), Custom CSS
*   **Machine Learning**: [Scikit-Learn](https://scikit-learn.org/) (Random Forest Regressor)
*   **Data Science**: Pandas, NumPy
*   **Model Management**: Joblib

---

## 📊 How It Works

1.  **Feature Engineering**: The model uses a 7-day lag window and rolling averages to capture price momentum.
2.  **Economic Impact**: It integrates the **USD-INR exchange rate**, as currency fluctuations directly impact precious metal pricing in India.
3.  **Recursive Forecasting**: To predict future days, the model feeds its own predictions back into the feature set for the next day.

---

## ▶️ Setup & Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model (Optional)
If you have new data in `data/final_data.csv`, run:
```bash
python main.py
```

### 3. Launch the Dashboard
```bash
streamlit run app.py
```

---

## 🔬 Insights & Analytics
The dashboard includes a dedicated **Insights Tab** featuring:
*   **USD-INR Exchange Rate Trend**: Visualizing the correlation between currency and metal value.
*   **Statistical Summaries**: Detailed breakdowns of min, max, median, and average prices.

---

## 👨‍💻 Author

**Alok Rana**
*   GitHub: [@alokrana01](https://github.com/alokrana01)

---

⭐ **If you find this project helpful, please give it a star!**
