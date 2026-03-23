from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf


@dataclass(slots=True)
class BacktestResult:
    ticker: str
    start_date: datetime
    end_date: datetime
    initial_cash: float
    final_strategy_value: float
    final_buy_hold_value: float
    total_trades: int
    buy_pct_threshold: float
    sell_pct_threshold: float
    buy_signals: list[tuple[pd.Timestamp, float]]
    sell_signals: list[tuple[pd.Timestamp, float]]
    data: pd.DataFrame

    @property
    def strategy_return_pct(self) -> float:
        return ((self.final_strategy_value - self.initial_cash) / self.initial_cash) * 100

    @property
    def buy_hold_return_pct(self) -> float:
        return ((self.final_buy_hold_value - self.initial_cash) / self.initial_cash) * 100


def fetch_price_history(ticker: str, start_years: int) -> tuple[pd.DataFrame, datetime, datetime]:
    end_date = datetime.now()
    start_date = end_date - timedelta(days=start_years * 365)
    data = yf.download(ticker, start=start_date, end=end_date, progress=False).copy()
    data = _normalize_price_data(data, ticker)
    return data, start_date, end_date


def _normalize_price_data(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if isinstance(data.columns, pd.MultiIndex):
        if "Price" in data.columns.names and "Ticker" in data.columns.names:
            data = data.xs(ticker, axis=1, level="Ticker")
        else:
            data = data.droplevel(-1, axis=1)

    return data.rename_axis(index=None, columns=None)


def backtest_strategy(
    ticker: str = "SPY",
    start_years: int = 20,
    initial_cash: float = 10000,
    buy_pct_threshold: float = .02,
    sell_pct_threshold: float = .02,
) -> BacktestResult:
    data, start_date, end_date = fetch_price_history(ticker=ticker, start_years=start_years)

    if data.empty:
        raise ValueError(
            f"No data found for {ticker}. Check the ticker symbol or your network connection."
        )

    data["Pct_Change"] = data["Close"].pct_change()

    cash = initial_cash
    shares = 0.0
    position = 0

    portfolio_values: list[float] = []
    buy_signals: list[tuple[pd.Timestamp, float]] = []
    sell_signals: list[tuple[pd.Timestamp, float]] = []

    for date, row in data.iterrows():
        price = float(row["Close"])
        change = row["Pct_Change"]

        if position == 0 and pd.notna(change) and change < -buy_pct_threshold:
            shares = cash / price
            cash = 0.0
            position = 1
            buy_signals.append((date, price))
        elif position == 1 and pd.notna(change) and change >= sell_pct_threshold:
            cash = shares * price
            shares = 0.0
            position = 0
            sell_signals.append((date, price))

        portfolio_values.append(cash + (shares * price))

    data["Strategy_Value"] = portfolio_values

    initial_price = float(data["Close"].iloc[0])
    buy_hold_shares = initial_cash / initial_price
    data["Buy_Hold_Value"] = data["Close"] * buy_hold_shares

    return BacktestResult(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        initial_cash=initial_cash,
        buy_pct_threshold=buy_pct_threshold,
        sell_pct_threshold=sell_pct_threshold,
        final_strategy_value=float(data["Strategy_Value"].iloc[-1]),
        final_buy_hold_value=float(data["Buy_Hold_Value"].iloc[-1]),
        total_trades=len(buy_signals) + len(sell_signals),
        buy_signals=buy_signals,
        sell_signals=sell_signals,
        data=data,
    )


def plot_backtest(result: BacktestResult) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(
        result.data.index,
        result.data["Buy_Hold_Value"],
        label=f"Buy & Hold ({result.ticker})",
        alpha=0.6,
    )
    plt.plot(
        result.data.index,
        result.data["Strategy_Value"],
        label=f"Strategy (+{result.sell_pct_threshold}% Sell / -{result.buy_pct_threshold}% Buy)",
        color="orange",
    )
    plt.title(f"{result.ticker} Strategy Backtest ({(result.end_date - result.start_date).days // 365} Years)")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
