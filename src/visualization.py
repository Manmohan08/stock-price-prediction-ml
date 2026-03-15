"""
Visualization Module for Stock Price Prediction
Generates professional charts for model comparison and analysis.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_all_results(df, results, backtest_results, feature_importances,
                     ticker, save_dir="results"):
    """Generate a comprehensive results dashboard."""
    os.makedirs(save_dir, exist_ok=True)

    # ---- Figure 1: Model Comparison ----
    plot_model_comparison(results, ticker, save_dir)

    # ---- Figure 2: Technical Indicators ----
    plot_technical_indicators(df, ticker, save_dir)

    # ---- Figure 3: Backtest Performance ----
    plot_backtest(backtest_results, ticker, save_dir)

    # ---- Figure 4: Feature Importance ----
    if feature_importances:
        plot_feature_importance(feature_importances, ticker, save_dir)


def plot_model_comparison(results, ticker, save_dir):
    """Bar chart comparing all model metrics."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle(f"Model Performance Comparison — {ticker}",
                 fontsize=15, fontweight='bold')

    models = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    titles = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']

    for i, (metric, title) in enumerate(zip(metrics, titles)):
        values = [results[m][metric] for m in models]
        bars = axes[i].barh(models, values, color=colors[i], edgecolor='white', height=0.6)
        axes[i].set_xlim(0, 1)
        axes[i].set_title(title, fontsize=12, fontweight='bold')
        axes[i].set_xlabel("Score")

        for bar, val in zip(bars, values):
            axes[i].text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                         f'{val:.3f}', va='center', fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(save_dir, "model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [✓] Saved: {path}")


def plot_technical_indicators(df, ticker, save_dir):
    """Plot key technical indicators on the stock data."""
    # Use last 200 data points for clarity
    df_plot = df.tail(200).copy()

    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
    fig.suptitle(f"Technical Indicators — {ticker}", fontsize=15, fontweight='bold')

    # Price with Bollinger Bands
    ax1 = axes[0]
    ax1.plot(df_plot.index, df_plot['Close'], color='#1E88E5', linewidth=1.5, label='Close')
    ax1.plot(df_plot.index, df_plot['SMA_20'], color='orange', linewidth=1, label='SMA 20', alpha=0.8)
    ax1.plot(df_plot.index, df_plot['SMA_50'], color='red', linewidth=1, label='SMA 50', alpha=0.8)
    if 'BB_Upper' in df_plot.columns:
        ax1.fill_between(df_plot.index, df_plot['BB_Upper'], df_plot['BB_Lower'],
                         alpha=0.1, color='blue', label='Bollinger Bands')
    ax1.set_ylabel("Price ($)")
    ax1.legend(loc='upper left', fontsize=8)
    ax1.set_title("Price & Moving Averages")
    ax1.grid(True, alpha=0.2)

    # RSI
    ax2 = axes[1]
    ax2.plot(df_plot.index, df_plot['RSI'], color='purple', linewidth=1.2)
    ax2.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Overbought (70)')
    ax2.axhline(y=30, color='green', linestyle='--', alpha=0.5, label='Oversold (30)')
    ax2.fill_between(df_plot.index, 30, 70, alpha=0.05, color='gray')
    ax2.set_ylabel("RSI")
    ax2.set_ylim(0, 100)
    ax2.legend(loc='upper left', fontsize=8)
    ax2.set_title("Relative Strength Index (RSI)")
    ax2.grid(True, alpha=0.2)

    # MACD
    ax3 = axes[2]
    ax3.plot(df_plot.index, df_plot['MACD'], color='#1E88E5', linewidth=1.2, label='MACD')
    ax3.plot(df_plot.index, df_plot['MACD_Signal'], color='orange', linewidth=1, label='Signal')
    colors_hist = ['#4CAF50' if v >= 0 else '#F44336' for v in df_plot['MACD_Hist']]
    ax3.bar(df_plot.index, df_plot['MACD_Hist'], color=colors_hist, alpha=0.5, width=1.5)
    ax3.set_ylabel("MACD")
    ax3.legend(loc='upper left', fontsize=8)
    ax3.set_title("MACD")
    ax3.grid(True, alpha=0.2)

    # Volume
    ax4 = axes[3]
    vol_colors = ['#4CAF50' if df_plot['Close'].iloc[i] >= df_plot['Open'].iloc[i]
                  else '#F44336' for i in range(len(df_plot))]
    ax4.bar(df_plot.index, df_plot['Volume'], color=vol_colors, alpha=0.6, width=1.5)
    ax4.set_ylabel("Volume")
    ax4.set_title("Trading Volume")
    ax4.grid(True, alpha=0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(save_dir, "technical_indicators.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [✓] Saved: {path}")


def plot_backtest(backtest_results, ticker, save_dir):
    """Plot portfolio value over time for each model's backtest."""
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle(f"Backtest: Portfolio Value Over Time — {ticker}",
                 fontsize=15, fontweight='bold')

    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0']
    for i, bt in enumerate(backtest_results):
        ax.plot(bt['portfolio_values'], label=f"{bt['model_name']} ({bt['total_return']:+.1f}%)",
                color=colors[i % len(colors)], linewidth=1.5)

    ax.axhline(y=backtest_results[0]['initial_capital'], color='gray',
               linestyle='--', alpha=0.5, label='Initial Capital')
    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(save_dir, "backtest_results.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [✓] Saved: {path}")


def plot_feature_importance(feature_importances, ticker, save_dir):
    """Plot feature importance from tree-based models."""
    fig, axes = plt.subplots(1, len(feature_importances), figsize=(14, 6))
    fig.suptitle(f"Feature Importance — {ticker}", fontsize=15, fontweight='bold')

    if len(feature_importances) == 1:
        axes = [axes]

    colors_map = {'Random Forest': '#2196F3', 'Gradient Boosting': '#4CAF50'}

    for ax, (model_name, importances) in zip(axes, feature_importances.items()):
        top_n = importances[:12]  # Top 12 features
        names = [x[0] for x in top_n][::-1]
        values = [x[1] for x in top_n][::-1]

        ax.barh(names, values, color=colors_map.get(model_name, '#FF9800'),
                edgecolor='white', height=0.6)
        ax.set_title(model_name, fontsize=12, fontweight='bold')
        ax.set_xlabel("Importance")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(save_dir, "feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [✓] Saved: {path}")
