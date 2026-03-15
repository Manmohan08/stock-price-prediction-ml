"""
Backtesting Engine for Stock Price Prediction
Simulates a basic trading strategy based on model predictions.
"""

import numpy as np


class Backtester:
    """
    Simulates trading based on ML model predictions and calculates
    performance metrics like total return, Sharpe ratio, and win rate.
    """

    def __init__(self, initial_capital=10000.0):
        self.initial_capital = initial_capital

    def run(self, df_test, predictions, model_name="Model"):
        """
        Run a backtest on test data using model predictions.

        Strategy:
        - If prediction = 1 (price goes up), BUY and hold for the day.
        - If prediction = 0 (price goes down), STAY OUT (hold cash).

        Args:
            df_test: DataFrame with at least 'Close' column for test period.
            predictions: Array of 0/1 predictions from the model.
            model_name: Name of the model for display.

        Returns:
            dict: Backtest results.
        """
        closes = df_test['Close'].values
        n = min(len(closes) - 1, len(predictions))

        capital = self.initial_capital
        shares = 0
        trades = 0
        wins = 0
        losses = 0
        portfolio_values = [capital]

        for i in range(n):
            daily_return = (closes[i + 1] - closes[i]) / closes[i]

            if predictions[i] == 1:  # Model says BUY
                trades += 1
                profit = capital * daily_return
                capital += profit
                if profit > 0:
                    wins += 1
                else:
                    losses += 1

            portfolio_values.append(capital)

        # Buy & Hold benchmark
        buy_hold_return = (closes[-1] - closes[0]) / closes[0] * 100
        buy_hold_final = self.initial_capital * (1 + buy_hold_return / 100)

        # Strategy metrics
        total_return = ((capital - self.initial_capital) / self.initial_capital) * 100
        win_rate = (wins / trades * 100) if trades > 0 else 0

        # Sharpe Ratio (annualized, assuming 252 trading days)
        daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        sharpe = 0.0
        if len(daily_returns) > 1 and np.std(daily_returns) > 0:
            sharpe = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252)

        # Max Drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdowns = (np.array(portfolio_values) - peak) / peak
        max_drawdown = np.min(drawdowns) * 100

        results = {
            'model_name': model_name,
            'initial_capital': self.initial_capital,
            'final_capital': capital,
            'total_return': total_return,
            'total_trades': trades,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'buy_hold_return': buy_hold_return,
            'buy_hold_final': buy_hold_final,
            'portfolio_values': portfolio_values,
        }

        return results

    @staticmethod
    def print_results(results):
        """Print formatted backtest results."""
        print(f"\n{'─' * 55}")
        print(f"  📈 BACKTEST: {results['model_name']}")
        print(f"{'─' * 55}")
        print(f"  Initial Capital:    ${results['initial_capital']:>12,.2f}")
        print(f"  Final Capital:      ${results['final_capital']:>12,.2f}")
        print(f"  Strategy Return:     {results['total_return']:>11.2f}%")
        print(f"  Buy & Hold Return:   {results['buy_hold_return']:>11.2f}%")
        print(f"  Total Trades:        {results['total_trades']:>11}")
        print(f"  Win Rate:            {results['win_rate']:>11.1f}%")
        print(f"  Sharpe Ratio:        {results['sharpe_ratio']:>11.2f}")
        print(f"  Max Drawdown:        {results['max_drawdown']:>11.2f}%")
        print(f"{'─' * 55}")
