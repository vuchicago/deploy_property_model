"""Public package API for stocks_analysis."""

from .backtest import BacktestResult, backtest_strategy, fetch_price_history, plot_backtest

__all__ = [
    "BacktestResult",
    "backtest_strategy",
    "fetch_price_history",
    "plot_backtest",
]
