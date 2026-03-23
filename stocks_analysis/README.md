# stocks-analysis

Simple stock backtesting utilities packaged as a Python library with a CLI.

## Setup

Create and activate the virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

If you already have the environment created, you can just activate it:

```bash
source .venv/bin/activate
```

## Run The Backtest

Run the CLI with the default settings:

```bash
stock-backtest
```

You can also run it without activating the environment:

```bash
.venv/bin/stock-backtest
```

## Available Arguments

```bash
stock-backtest --help
```

Arguments:

- `--ticker`: Ticker symbol to backtest. Default: `SPY`
- `--start-years`: Number of years of price history to analyze. Default: `20`
- `--buy-pct-threshold`: what percent threshold would you buy stocks (sell if <2%, buy if drop 2%).  Default `.02`
- `--sell-pct-threshold`: what percent threshold would you sell stocks (sell if <2%, buy if drop 2%).  Default `.02`
- `--initial-cash`: Starting portfolio cash. Default: `10000`
- `--no-plot`: Skip the chart and only print the summary results

## Examples

Run the default SPY backtest:

```bash
stock-backtest
```

Run 10 years of Apple data with a $5,000 starting balance:

```bash
stock-backtest --ticker AAPL --start-years 10 --initial-cash 5000 --buy-pct-threshold .5 --sell-pct-threshold .01
```

Run Tesla without opening a plot:

```bash
stock-backtest --ticker TSLA --start-years 5 --no-plot
```

## Import As A Library

```python
from stocks_analysis import backtest_strategy, plot_backtest

result = backtest_strategy(ticker="SPY", start_years=10, initial_cash=10000)
print(result.final_strategy_value)
plot_backtest(result)
```
