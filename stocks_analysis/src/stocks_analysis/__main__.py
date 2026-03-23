from __future__ import annotations

import argparse

from .backtest import backtest_strategy, plot_backtest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the stock backtest strategy.")
    parser.add_argument("--ticker", default="SPY", help="Ticker symbol to backtest.")
    parser.add_argument(
        "--start-years",
        type=int,
        default=20,
        help="How many years of history to analyze.",
    )
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=10000,
        help="Initial cash balance for the backtest.",
    )
    parser.add_argument(
        "--buy-pct-threshold",
        type=float,
        default=.02,
        help="what is the pct threshold to buy your stock",      
    )
    parser.add_argument(
        "--sell-pct-threshold",
        type=float,
        default=.02,
        help="what is the pct threshold to d sell your stock",      
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting the strategy and buy-and-hold values.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    print(
        f"Downloading data for {args.ticker} from the last {args.start_years} year(s)..."
    )
    result = backtest_strategy(
        ticker=args.ticker,
        start_years=args.start_years,
        initial_cash=args.initial_cash,
        buy_pct_threshold=args.buy_pct_threshold,
        sell_pct_threshold=args.sell_pct_threshold
    )

    print("-" * 40)
    print(f"Initial Investment: ${result.initial_cash:,.2f}")
    print(f"Final Strategy Value: ${result.final_strategy_value:,.2f}")
    print(f"Final Buy & Hold Value: ${result.final_buy_hold_value:,.2f}")
    print("-" * 40)
    print(f"Strategy Return: {result.strategy_return_pct:.2f}%")
    print(f"Buy & Hold Return: {result.buy_hold_return_pct:.2f}%")
    print("-" * 40)
    print(f"Total Trades: {result.total_trades}")

    if not args.no_plot:
        plot_backtest(result)


if __name__ == "__main__":
    main()
