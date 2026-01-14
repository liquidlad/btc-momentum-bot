"""Simple optimization script."""
import sys
import os

# Set up path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'strategy'))

from backtest import Backtester, DEFAULT_PARAMS
import pandas as pd
import itertools
import json

print('Loading data...')
df = pd.read_csv('data/BTCUSDT_1m.csv', parse_dates=['open_time'], index_col='open_time')
print(f'Loaded {len(df)} candles')

# Smaller parameter ranges for faster testing
param_ranges = {
    'ema_fast': [5, 9],
    'ema_slow': [15, 21],
    'roc_threshold': [0.08, 0.12],
    'volume_multiplier': [1.2, 1.5],
    'stop_loss_pct': [0.12, 0.15],
    'take_profit_1_pct': [0.10, 0.15],
}

# Generate combinations
keys = list(param_ranges.keys())
values = list(param_ranges.values())
combinations = list(itertools.product(*values))
print(f'Testing {len(combinations)} combinations...')

results = []
for i, combo in enumerate(combinations):
    params = DEFAULT_PARAMS.copy()
    param_dict = dict(zip(keys, combo))
    params.update(param_dict)

    try:
        bt = Backtester(params, capital=200, leverage=50)
        metrics = bt.run(df)

        if 'error' not in metrics:
            # Calculate score prioritizing volume
            volume_score = min(metrics['trades_per_day'] / 100, 1.0)
            profit_score = min(metrics['total_return_pct'] / 10, 1.0) if metrics['total_return_pct'] > 0 else 0
            risk_score = max(1 - (metrics['max_drawdown_pct'] / 25), 0)

            score = (volume_score * 0.5) + (profit_score * 0.25) + (risk_score * 0.25)
            metrics['score'] = score
            metrics['params'] = param_dict
            results.append(metrics)
            print(f"  [{i+1}/{len(combinations)}] Score: {score:.3f}, Trades/day: {metrics['trades_per_day']:.1f}, Return: {metrics['total_return_pct']:.1f}%, DD: {metrics['max_drawdown_pct']:.1f}%")
    except Exception as e:
        print(f"  [{i+1}] Error: {e}")

# Sort by score
results.sort(key=lambda x: x['score'], reverse=True)

print('\n' + '='*60)
print('TOP 5 CONFIGURATIONS')
print('='*60)

for i, r in enumerate(results[:5]):
    print(f"\n#{i+1} - Score: {r['score']:.4f}")
    print(f"  Trades/Day: {r['trades_per_day']:.1f}")
    print(f"  Win Rate: {r['win_rate']:.1f}%")
    print(f"  Total Return: {r['total_return_pct']:.2f}%")
    print(f"  Max DD: {r['max_drawdown_pct']:.2f}%")
    print(f"  Profit Factor: {r['profit_factor']:.2f}")
    print(f"  Params: {r['params']}")

# Save best params
if results:
    best = results[0]
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
    print('\nSaved best config to config/optimized_params.json')
