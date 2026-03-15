"""
Advanced Data Handler for Stock Price Prediction
Fetches stock data and engineers powerful technical indicators for ML models.
"""

import yfinance as yf
import pandas as pd
import numpy as np


def fetch_stock_data(ticker, start_date, end_date):
    """Fetch historical stock data using yfinance."""
    print(f"📥 Fetching data for {ticker} ({start_date} to {end_date})...")
    data = yf.download(ticker, start=start_date, end=end_date)

    # Flatten multi-level columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data


def add_technical_indicators(df):
    """
    Engineer advanced technical indicators as ML features.

    Indicators added:
    - SMA (10, 20, 50)     — Simple Moving Averages
    - EMA (12, 26)         — Exponential Moving Averages
    - RSI (14)             — Relative Strength Index
    - MACD + Signal Line   — Moving Average Convergence Divergence
    - Bollinger Bands      — Upper, Middle, Lower bands + Band Width
    - ATR (14)             — Average True Range (volatility)
    - OBV                  — On-Balance Volume
    - Price Rate of Change — Momentum indicator
    - Stochastic Oscillator — %K and %D
    """
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']

    # ---- Moving Averages ----
    df['SMA_10'] = close.rolling(window=10).mean()
    df['SMA_20'] = close.rolling(window=20).mean()
    df['SMA_50'] = close.rolling(window=50).mean()
    df['EMA_12'] = close.ewm(span=12, adjust=False).mean()
    df['EMA_26'] = close.ewm(span=26, adjust=False).mean()

    # ---- RSI (Relative Strength Index) ----
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))

    # ---- MACD ----
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # ---- Bollinger Bands ----
    bb_period = 20
    df['BB_Middle'] = close.rolling(window=bb_period).mean()
    bb_std = close.rolling(window=bb_period).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / (df['BB_Middle'] + 1e-10)

    # ---- ATR (Average True Range) ----
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()

    # ---- On-Balance Volume (OBV) ----
    obv = [0]
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i - 1]:
            obv.append(obv[-1] + volume.iloc[i])
        elif close.iloc[i] < close.iloc[i - 1]:
            obv.append(obv[-1] - volume.iloc[i])
        else:
            obv.append(obv[-1])
    df['OBV'] = obv

    # ---- Price Rate of Change (ROC) ----
    df['ROC'] = close.pct_change(periods=10) * 100

    # ---- Stochastic Oscillator ----
    low_14 = low.rolling(window=14).min()
    high_14 = high.rolling(window=14).max()
    df['Stoch_K'] = ((close - low_14) / (high_14 - low_14 + 1e-10)) * 100
    df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()

    # ---- Daily Returns & Volatility ----
    df['Daily_Return'] = close.pct_change()
    df['Volatility_10'] = df['Daily_Return'].rolling(window=10).std()

    return df


def prepare_dataset(df):
    """
    Prepare the final ML-ready dataset.
    Creates target labels and drops NaN rows.
    """
    # Target: 1 if next day close > today's close, else 0
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    # Drop rows with NaN values
    df.dropna(inplace=True)

    # Feature columns (exclude raw OHLCV and target)
    feature_cols = [
        'SMA_10', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
        'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
        'BB_Upper', 'BB_Lower', 'BB_Width',
        'ATR', 'OBV', 'ROC',
        'Stoch_K', 'Stoch_D',
        'Daily_Return', 'Volatility_10',
        'Open', 'High', 'Low', 'Close', 'Volume'
    ]

    return df, feature_cols
