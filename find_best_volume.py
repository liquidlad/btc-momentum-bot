"""
Find best balance between volume and profitability.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'strategy'))

from backtest import Backtester, DEFAULT_PARAMS
import pandas as pd
import itertools
import json

print('Loading data...')
df = pd.read_csv('data/BTCUSDT_1m.csv', parse_dates=['open_time'], index_col='open_time')
print(f'Loaded {len(df)} candles')

# Wider search including min_conditions=2
param_ranges = {
    'min_conditions': [2, 3],
    'ema_fast': [3, 5, 7],
    'ema_slow': [10, 15, 20],
    'roc_threshold': [0.05, 0.08, 0.10],
    'volume_multiplier': [1.0, 1.2],
    'stop_loss_pct': [0.10, 0.12, 0.15],
    'take_profit_1_pct': [0.08, 0.10, 0.12],
}

base_params = DEFAULT_PARAMS.copy()
base_params['max_candles_in_trade'] = 8
base_params['profit_check_candles'] = 4
base_params['rsi_long_min'] = 25
base_params['rsi_long_max'] = 75
base_params['rsi_short_min'] = 25
base_params['rsi_short_max'] = 75

keys = list(param_ranges.keys())
values = list(param_ranges.values())
combinations = list(itertools.product(*values))
print(f'Testing {len(combinations)} combinations...')

results = []
for i, combo in enumerate(combinations):
    params = base_params.copy()
    param_dict = dict(zip(keys, combo))
    params.update(param_dict)

    try:
        bt = Backtester(params, capital=200, leverage=50)
        metrics = bt.run(df)

        # Only keep profitable configs
        if 'error' not in metrics and metrics.get('total_return_pct', 0) > 5:  # At least 5% return
            trades_day = metrics['trades_per_day']
            ret = metrics['total_return_pct']
            dd = metrics['max_drawdown_pct']

            # Score balancing volume and safety
            volume_score = min(trades_day / 80, 1.0)  # Target 80/day
            profit_score = min(ret / 30, 1.0)  # Target 30% return
            risk_score = max(1 - (dd / 25), 0)  # Under 25% DD

            score = (volume_score * 0.5) + (profit_score * 0.3) + (risk_score * 0.2)
            metrics['score'] = score
            metrics['params'] = param_dict
            results.append(metrics)
    except:
        pass

    if (i + 1) % 100 == 0:
        print(f'Progress: {i+1}/{len(combinations)}')

results.sort(key=lambda x: x['score'], reverse=True)

print('\n' + '='*60)
print('TOP 10 PROFITABLE HIGH-VOLUME CONFIGURATIONS')
print('='*60)

for i, r in enumerate(results[:10]):
    print(f"\n#{i+1} - Score: {r['score']:.4f}")
    print(f"  Trades/Day: {r['trades_per_day']:.1f}")
    print(f"  Win Rate: {r['win_rate']:.1f}%")
    print(f"  Return: {r['total_return_pct']:.2f}%")
    print(f"  Max DD: {r['max_drawdown_pct']:.2f}%")
    print(f"  Profit Factor: {r['profit_factor']:.2f}")
    print(f"  Params: {r['params']}")

# Select best meeting criteria
best = None
for r in results:
    if r['total_return_pct'] > 10 and r['max_drawdown_pct'] < 30:
        best = r
        break

if not best and results:
    best = results[0]

if best:
    optimized = {
        'optimized_params': best['params'],
        'expected_metrics': {
            'trades_per_day': best['trades_per_day'],
            'win_rate': best['win_rate'],
            'profit_factor': best['profit_factor'],
            'total_return_pct': best['total_return_pct'],
            'max_drawdown_pct': best['max_drawdown_pct'],
        }
    }
    with open('config/optimized_params.json', 'w') as f:
        json.dump(optimized, f, indent=2)

    daily_volume = best['trades_per_day'] * 2000  # Assuming $2000 avg position
    print(f"\nBEST CONFIG SAVED:")
    print(f"  Trades/day: {best['trades_per_day']:.1f}")
    print(f"  Est. daily volume: ${daily_volume:,.0f} (at $2000/trade)")
    print(f"  Return over 90 days: {best['total_return_pct']:.2f}%")
    print(f"  Max drawdown: {best['max_drawdown_pct']:.2f}%")
