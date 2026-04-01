# 🪙 Metal Intelligence: Gold & Silver Price Prediction

> **Predict the future of precious metals with Machine Learning.**

Welcome to **Metal Intelligence**, a comprehensive data science project designed to track, analyze, and forecast the prices of Gold (24K, 22K) and Silver. Using historical data, technical indicators, and machine learning models, this dashboard provides actionable insights into market trends.

---

## 🚀 Key Features

-   **📈 Real-time Data**: Fetches latest market data for Gold and Silver using `yfinance`.
-   **📅 7-Day Forecasting**: High-accuracy predictions for the upcoming week.
-   **📊 Interactive Dashboard**: Built with **Streamlit** for a seamless user experience.
-   **📉 Technical Indicators**: Uses Moving Averages (7 & 30 days) and Lag features for precision.
-   **💱 Currency Tracking**: Monitors USD-INR exchange rates to understand local price impacts.
-   **📄 Exportable Reports**: Download prediction results as CSV for further analysis.

---

## 🛠️ Tech Stack

| Category | Tools |
| :--- | :--- |
| **Language** | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) |
| **Frontend** | ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white) |
| **Data Processing** | ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white) |
| **Machine Learning** | ![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white) |
| **Visualization** | ![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white) |

---

## 📦 Project Structure

```bash
Gold_Silver_Prediction/
├── app.py              # Streamlit Web Application
├── main.py             # Full Pipeline (Fetch -> Train -> Predict)
├── banner.png          # Project Banner
├── requirements.txt    # Python Dependencies
├── data/               # Raw and Processed Datasets
├── models/             # Serialized ML Models (.pkl)
├── notebooks/          # Exploratory Data Analysis
└── src/                # Core Source Code
    ├── data/           # Data Fetching Scripts
    ├── processing/     # Data Preprocessing & Cleaning
    └── models/         # Training and Prediction Logic
```

---

## ⚙️ Installation & Setup

### 1. Clone the Project
```bash
git clone https://github.com/alokrana01/Gold_Silver_Prediction.git
cd Gold_Silver_Prediction
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🏃 Usage

### Run the Full Pipeline
To fetch the latest data and retrain the models:
```bash
python main.py
```

### Launch the Dashboard
To see the interactive charts and forecasts:
```bash
streamlit run app.py
```

---

## 🧠 Machine Learning Model

The project uses a **Random Forest Regressor** (via Scikit-Learn) trained on several key features:
- **Lag Features**: Prices from the previous 1, 2, and 3 days.
- **Moving Averages**: 7-day and 30-day trends.
- **Day of Week**: Captures weekly seasonality.
- **Currency Impact**: Incorporates USD-INR fluctuations.

---

## 🤝 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---

**Developed with ❤️ by [Alok Rana](https://github.com/alok-rana-7)**

> *Disclaimer: This tool is for educational purposes only. Financial investments carry risks.*
