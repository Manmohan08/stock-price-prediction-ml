"""
Stock Market Prediction using Machine Learning — v2.0

A comprehensive stock price direction prediction system that:
1. Fetches real-time stock data from Yahoo Finance
2. Engineers 15+ technical indicators as ML features
3. Trains and compares 5 different ML models
4. Backtests predictions with a simulated trading strategy
5. Generates professional visualization dashboards

Author: Manmohan Kumar
"""

from src.data_handler import fetch_stock_data, add_technical_indicators, prepare_dataset
from src.model import ModelPipeline
from src.backtester import Backtester
from src.visualization import plot_all_results


def main():
    # ======== CONFIGURATION ========
    TICKER = 'AAPL'              # Stock ticker symbol
    START_DATE = '2019-01-01'    # Start of data range
    END_DATE = '2024-12-31'      # End of data range
    INITIAL_CAPITAL = 10000      # Starting capital for backtesting ($)
    TEST_RATIO = 0.2             # 80% train, 20% test

    print("\n" + "=" * 60)
    print("  📈  STOCK MARKET PREDICTION — v2.0")
    print("=" * 60)
    print(f"  Ticker:         {TICKER}")
    print(f"  Date Range:     {START_DATE} → {END_DATE}")
    print(f"  Backtest Capital: ${INITIAL_CAPITAL:,}")
    print("=" * 60)

    # ======== STEP 1: FETCH DATA ========
    raw_data = fetch_stock_data(TICKER, START_DATE, END_DATE)
    if raw_data.empty:
        print("❌ Failed to fetch data. Check your ticker and internet connection.")
        return
    print(f"✅ Data fetched: {len(raw_data)} trading days\n")

    # ======== STEP 2: FEATURE ENGINEERING ========
    print("🔧 Engineering technical indicators...")
    data_with_indicators = add_technical_indicators(raw_data.copy())
    processed_data, feature_cols = prepare_dataset(data_with_indicators)
    print(f"✅ Features engineered: {len(feature_cols)} indicators")
    print(f"✅ Clean dataset: {len(processed_data)} samples\n")

    # ======== STEP 3: TRAIN & EVALUATE MODELS ========
    pipeline = ModelPipeline()
    results, y_test, X_test = pipeline.train_and_evaluate(
        processed_data, feature_cols, test_ratio=TEST_RATIO
    )

    # ======== STEP 4: BACKTESTING ========
    print("\n" + "=" * 60)
    print("  💰  BACKTESTING TRADING STRATEGY")
    print("=" * 60)

    test_start = int(len(processed_data) * (1 - TEST_RATIO))
    df_test = processed_data.iloc[test_start:].copy()

    backtester = Backtester(initial_capital=INITIAL_CAPITAL)
    backtest_results = []

    for model_name, model_metrics in results.items():
        bt = backtester.run(df_test, model_metrics['predictions'], model_name)
        backtester.print_results(bt)
        backtest_results.append(bt)

    # ======== STEP 5: FEATURE IMPORTANCE ========
    feature_importances = pipeline.get_feature_importance(feature_cols)

    print("\n🔑 Top 5 Most Important Features:")
    for model_name, importances in feature_importances.items():
        print(f"\n  {model_name}:")
        for feat, imp in importances[:5]:
            print(f"    • {feat:<20} {imp:.4f}")

    # ======== STEP 6: GENERATE CHARTS ========
    print("\n📊 Generating visualizations...")
    plot_all_results(processed_data, results, backtest_results,
                     feature_importances, TICKER)

    # ======== FINAL SUMMARY ========
    best_model = max(results, key=lambda k: results[k]['f1_score'])
    best_bt = next(bt for bt in backtest_results if bt['model_name'] == best_model)

    print("\n" + "=" * 60)
    print("  🏆  FINAL SUMMARY")
    print("=" * 60)
    print(f"  Best Model:      {best_model}")
    print(f"  F1-Score:        {results[best_model]['f1_score']:.4f}")
    print(f"  Strategy Return: {best_bt['total_return']:+.2f}%")
    print(f"  Win Rate:        {best_bt['win_rate']:.1f}%")
    print(f"  Sharpe Ratio:    {best_bt['sharpe_ratio']:.2f}")
    print("=" * 60)
    print("\n✅ All results saved to the 'results/' folder.")
    print("🚀 Prediction complete!\n")


if __name__ == "__main__":
    main()
