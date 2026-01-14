"""
Volume-focused optimization - reduce entry requirements to generate more trades.
Target: 100+ trades/day while maintaining profitability.
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

# More aggressive parameters to generate volume
param_ranges = {
    'ema_fast': [3, 5],           # Faster EMAs
    'ema_slow': [8, 12],          # Faster slow EMA
    'roc_threshold': [0.03, 0.05, 0.07],  # Lower threshold = more signals
    'volume_multiplier': [1.0, 1.1, 1.2],  # Lower volume requirement
    'stop_loss_pct': [0.08, 0.10],  # Tighter stops
    'take_profit_1_pct': [0.05, 0.08],  # Quicker profits
    'max_candles_in_trade': [5, 8],  # Shorter holds
}

# Also reduce min_conditions if needed
base_params = DEFAULT_PARAMS.copy()
base_params['profit_check_candles'] = 3  # Exit faster if not profitable
base_params['rsi_long_min'] = 30  # Wider RSI bands
base_params['rsi_long_max'] = 70
base_params['rsi_short_min'] = 30
base_params['rsi_short_max'] = 70

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

        if 'error' not in metrics and metrics['total_pnl'] > 0:
            # Score heavily weighted toward volume
            volume_score = min(metrics['trades_per_day'] / 100, 1.0)
            profit_score = min(metrics['total_return_pct'] / 20, 1.0) if metrics['total_return_pct'] > 0 else 0
            risk_score = max(1 - (metrics['max_drawdown_pct'] / 25), 0)

            # 60% volume, 20% profit, 20% risk
            score = (volume_score * 0.6) + (profit_score * 0.2) + (risk_score * 0.2)
            metrics['score'] = score
            metrics['params'] = param_dict
            results.append(metrics)

            if metrics['trades_per_day'] > 50:  # Only show high-volume configs
                print(f"  [{i+1}] Score: {score:.3f}, Trades/day: {metrics['trades_per_day']:.1f}, Return: {metrics['total_return_pct']:.1f}%, DD: {metrics['max_drawdown_pct']:.1f}%")
    except Exception as e:
        pass

    if (i + 1) % 50 == 0:
        print(f'Progress: {i+1}/{len(combinations)}')

# Sort by score
results.sort(key=lambda x: x['score'], reverse=True)

print('\n' + '='*60)
print('TOP 10 HIGH-VOLUME CONFIGURATIONS')
print('='*60)

for i, r in enumerate(results[:10]):
    print(f"\n#{i+1} - Score: {r['score']:.4f}")
    print(f"  Trades/Day: {r['trades_per_day']:.1f}")
    print(f"  Win Rate: {r['win_rate']:.1f}%")
    print(f"  Total Return: {r['total_return_pct']:.2f}%")
    print(f"  Max DD: {r['max_drawdown_pct']:.2f}%")
    print(f"  Profit Factor: {r['profit_factor']:.2f}")
    print(f"  Params: {r['params']}")

# Save best high-volume config
if results:
    # Find best that has at least 50 trades/day
    high_vol_results = [r for r in results if r['trades_per_day'] >= 50]
    if high_vol_results:
        best = high_vol_results[0]
    else:
        best = results[0]

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
    print(f'\nSaved config with {best["trades_per_day"]:.1f} trades/day to config/optimized_params.json')
