# 📈 Stock Market Prediction using Machine Learning — v2.0

An advanced stock price prediction system that uses **15+ technical indicators** and **5 ML algorithms** to predict next-day price direction. Includes backtesting and professional visualizations.

## 🚀 Features

- **15+ Technical Indicators** — SMA, EMA, RSI, MACD, Bollinger Bands, ATR, OBV, Stochastic Oscillator, and more
- **5 ML Models Compared** — Random Forest, Gradient Boosting, SVM, KNN, Logistic Regression
- **Backtesting Engine** — Simulated trading strategy with Sharpe Ratio, Win Rate, and Max Drawdown
- **Professional Charts** — Model comparison, technical indicators, portfolio curves, feature importance
- **Real-Time Data** — Fetches live data from Yahoo Finance via `yfinance`

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Core language |
| Pandas & NumPy | Data manipulation |
| Scikit-learn | ML model training & evaluation |
| yfinance | Stock data fetching |
| Matplotlib | Chart generation |

## 📂 Project Structure

```
stock-price-prediction-ml/
├── README.md
├── requirements.txt
├── .gitignore
├── main.py                  # Main prediction pipeline
├── results/                 # Generated charts & plots
│   ├── model_comparison.png
│   ├── technical_indicators.png
│   ├── backtest_results.png
│   └── feature_importance.png
└── src/
    ├── __init__.py
    ├── data_handler.py      # Data fetching & indicator engineering
    ├── model.py             # Multi-model ML pipeline
    ├── backtester.py        # Trading strategy backtester
    └── visualization.py     # Chart generation
```

## ⚙️ Setup & Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Manmohan08/stock-price-prediction-ml.git
   cd stock-price-prediction-ml
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the prediction:**
   ```bash
   python main.py
   ```

## 📊 Sample Output

```
══════════════════════════════════════════════════════════════
  📈  STOCK MARKET PREDICTION — v2.0
══════════════════════════════════════════════════════════════
  Ticker:         AAPL
  Date Range:     2019-01-01 → 2024-12-31

📋  MODEL COMPARISON
══════════════════════════════════════════════════════════════
  Model                      Accuracy  Precision     Recall         F1
  Random Forest                0.XXXX     0.XXXX     0.XXXX     0.XXXX
  Gradient Boosting            0.XXXX     0.XXXX     0.XXXX     0.XXXX
  SVM (RBF)                    0.XXXX     0.XXXX     0.XXXX     0.XXXX
  KNN                          0.XXXX     0.XXXX     0.XXXX     0.XXXX
  Logistic Regression          0.XXXX     0.XXXX     0.XXXX     0.XXXX

🏆  Best Model: [Model Name] (F1-Score: 0.XXXX)
```

Charts are saved automatically to the `results/` folder.

## 📈 Technical Indicators Used

| Indicator | Description |
|-----------|-------------|
| SMA (10, 20, 50) | Simple Moving Averages |
| EMA (12, 26) | Exponential Moving Averages |
| RSI (14) | Relative Strength Index |
| MACD | Moving Average Convergence Divergence |
| Bollinger Bands | Price volatility bands |
| ATR (14) | Average True Range |
| OBV | On-Balance Volume |
| Stochastic %K/%D | Momentum oscillator |
| ROC | Rate of Change |

## ⚠️ Disclaimer

This project is for **educational purposes only**. Stock market predictions are inherently uncertain. Do not use this model for real trading decisions without proper risk management.

## 👤 Author

**Manmohan Kumar**

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
