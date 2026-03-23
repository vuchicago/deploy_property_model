#%%
from stocks_analysis import backtest_strategy, plot_backtest


initial_cash=10000
start_years=30
buy_pct_threshold=.005
sell_pct_threshold=.1
ticker="SPY"


result = backtest_strategy(ticker=ticker, start_years=start_years, initial_cash=initial_cash, buy_pct_threshold=buy_pct_threshold,sell_pct_threshold=sell_pct_threshold)

print('buy and hold: ', result.final_buy_hold_value)
print('buy and hold pct: ',result.buy_hold_return_pct)
print('strategy: ', result.final_strategy_value)
print('strategy pct: ', result.buy_hold_return_pct)
plot_backtest(result)
# %%
