import pandas as pd
from src.data_handler import fetch_stock_data, preprocess_data
from src.model import train_and_evaluate

def main():
    # Parameters
    ticker = 'AAPL'  # Apple Inc. as an example
    start_date = '2020-01-01'
    end_date = '2024-01-01'
    
    # 1. Fetch Data
    raw_data = fetch_stock_data(ticker, start_date, end_date)
    
    if raw_data.empty:
        print("Failed to fetch data.")
        return
        
    print(f"Data fetched successfully. Total rows: {len(raw_data)}")
    
    # 2. Preprocess Data
    processed_data = preprocess_data(raw_data)
    print(f"Data preprocessed successfully. Remaining rows for training: {len(processed_data)}")
    
    # 3. Train Model and Evaluate
    model = train_and_evaluate(processed_data)
    print("Prediction process completed successfully.")

if __name__ == "__main__":
    main()
