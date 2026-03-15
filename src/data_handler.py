import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetches historical stock data using yfinance.
    """
    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def preprocess_data(df):
    """
    Prepares the data for basic machine learning modeling.
    Creates moving averages and target labels.
    """
    # Create simple moving averages
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # Target: 1 if Next Day Close > Today's Close, else 0
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # Drop NaNs created by rolling and shifting
    df.dropna(inplace=True)
    return df
