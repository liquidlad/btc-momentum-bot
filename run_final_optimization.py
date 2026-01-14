"""
Final optimization with min_conditions parameter to maximize volume.
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

# Parameters including min_conditions
param_ranges = {
    'min_conditions': [2, 3, 4],  # KEY: require fewer conditions
    'ema_fast': [3, 5],
    'ema_slow': [8, 12],
    'roc_threshold': [0.03, 0.05],
    'volume_multiplier': [1.0, 1.2],
    'stop_loss_pct': [0.08, 0.10],
    'take_profit_1_pct': [0.06, 0.08],
    'max_candles_in_trade': [5, 8],
}

base_params = DEFAULT_PARAMS.copy()
base_params['profit_check_candles'] = 3
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

        if 'error' not in metrics and metrics.get('total_pnl', 0) > 0:
            # Score heavily weighted toward volume (target 100+/day)
            volume_score = min(metrics['trades_per_day'] / 100, 1.0)
            profit_score = min(metrics['total_return_pct'] / 10, 1.0) if metrics['total_return_pct'] > 0 else 0
            risk_score = max(1 - (metrics['max_drawdown_pct'] / 25), 0)

            score = (volume_score * 0.6) + (profit_score * 0.2) + (risk_score * 0.2)
            metrics['score'] = score
            metrics['params'] = param_dict
            results.append(metrics)

            if metrics['trades_per_day'] > 80:
                print(f"  [{i+1}] Score: {score:.3f}, Trades/day: {metrics['trades_per_day']:.1f}, Return: {metrics['total_return_pct']:.1f}%, DD: {metrics['max_drawdown_pct']:.1f}%")
    except Exception as e:
        pass

    if (i + 1) % 100 == 0:
        print(f'Progress: {i+1}/{len(combinations)}')

# Sort by score
results.sort(key=lambda x: x['score'], reverse=True)

print('\n' + '='*60)
print('TOP 10 CONFIGURATIONS')
print('='*60)

for i, r in enumerate(results[:10]):
    print(f"\n#{i+1} - Score: {r['score']:.4f}")
    print(f"  Trades/Day: {r['trades_per_day']:.1f}")
    print(f"  Win Rate: {r['win_rate']:.1f}%")
    print(f"  Total Return: {r['total_return_pct']:.2f}%")
    print(f"  Max DD: {r['max_drawdown_pct']:.2f}%")
    print(f"  Profit Factor: {r['profit_factor']:.2f}")
    print(f"  Params: {r['params']}")

# Find best config meeting volume target
high_vol = [r for r in results if r['trades_per_day'] >= 100]
if high_vol:
    best = high_vol[0]
    print(f"\n*** Found {len(high_vol)} configs with 100+ trades/day ***")
else:
    # Take highest volume profitable config
    profitable = [r for r in results if r['total_return_pct'] > 0]
    profitable.sort(key=lambda x: x['trades_per_day'], reverse=True)
    best = profitable[0] if profitable else results[0]
    print(f"\n*** No 100+ trades/day config found, using highest volume: {best['trades_per_day']:.1f}/day ***")

# Save best
optimized = {
    'optimized_params': best['params'],
    'expected_metrics': {
        'trades_per_day': best['trades_per_day'],
        'win_rate': best['win_rate'],
        'profit_factor': best['profit_factor'],
        'total_return_pct': best['total_return_pct'],
        'max_drawdown_pct': best['max_drawdown_pct'],
        'score': best['score'],
    }
}
with open('config/optimized_params.json', 'w') as f:
    json.dump(optimized, f, indent=2)
print(f'\nSaved config to config/optimized_params.json')
print(f'Expected: {best["trades_per_day"]:.1f} trades/day, {best["total_return_pct"]:.2f}% return over 90 days')
