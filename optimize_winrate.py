"""
Optimizer focused on high win rate and ROI with minimum 5 trades/day.
"""

import json
import itertools
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'strategy'))
from backtest import Backtester, DEFAULT_PARAMS


def optimize_for_winrate(data_path: str, min_trades_per_day: float = 5.0):
    """
    Optimize for highest ROI with minimum trades/day constraint.
    """
    # Load data
    df = pd.read_csv(data_path, parse_dates=["open_time"], index_col="open_time")
    trading_days = (df.index[-1] - df.index[0]).days or 1

    print(f"Data: {len(df)} candles over {trading_days} days")
    print(f"Constraint: >= {min_trades_per_day} trades/day")
    print()

    # Parameter ranges for exploration
    optimization_ranges = {
        "min_conditions": [2, 3],
        "ema_fast": [3, 5],
        "ema_slow": [15, 21],
        "roc_threshold": [0.05, 0.10],
        "volume_multiplier": [1.0],
        "stop_loss_pct": [0.30, 0.50],
        "take_profit_1_pct": [0.50, 0.80, 1.0],
        "max_hold_candles": [15, 30],
    }

    # Generate combinations
    keys = list(optimization_ranges.keys())
    values = list(optimization_ranges.values())
    combinations = list(itertools.product(*values))

    print(f"Testing {len(combinations)} parameter combinations...")
    print()

    results = []

    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))

        # Skip invalid combinations (ema_fast must be < ema_slow)
        if params["ema_fast"] >= params["ema_slow"]:
            continue

        # Merge with defaults
        test_params = DEFAULT_PARAMS.copy()
        test_params.update(params)

        try:
            bt = Backtester(test_params, capital=200, leverage=50)
            metrics = bt.run(df)

            if "error" in metrics:
                continue

            trades_per_day = metrics.get("trades_per_day", 0)

            # Filter: minimum trades per day
            if trades_per_day < min_trades_per_day:
                continue

            metrics["params"] = params
            results.append(metrics)

        except Exception as e:
            continue

        if (i + 1) % 500 == 0:
            print(f"Progress: {i + 1}/{len(combinations)} - Found {len(results)} valid configs", end="\r")

    print(f"\nFound {len(results)} configurations with >= {min_trades_per_day} trades/day")
    print()

    if not results:
        print("No configurations met the criteria!")
        return

    # Sort by ROI (total_return_pct), then by win_rate
    results.sort(key=lambda x: (x["total_return_pct"], x["win_rate"]), reverse=True)

    # Show top 20
    print("=" * 80)
    print("TOP 20 CONFIGURATIONS (sorted by ROI)")
    print("=" * 80)
    print(f"{'#':<3} {'ROI%':<10} {'WinRate':<10} {'Trades/Day':<12} {'MaxDD%':<10} {'PF':<8}")
    print("-" * 80)

    for i, r in enumerate(results[:20]):
        print(f"{i+1:<3} {r['total_return_pct']:<10.2f} {r['win_rate']:<10.1f} {r['trades_per_day']:<12.1f} {r['max_drawdown_pct']:<10.2f} {r['profit_factor']:<8.2f}")

    print()
    print("=" * 80)
    print("BEST CONFIGURATION DETAILS")
    print("=" * 80)

    best = results[0]
    print(f"Total Return: {best['total_return_pct']:.2f}%")
    print(f"Win Rate: {best['win_rate']:.1f}%")
    print(f"Trades/Day: {best['trades_per_day']:.1f}")
    print(f"Total Trades: {best['total_trades']}")
    print(f"Max Drawdown: {best['max_drawdown_pct']:.2f}%")
    print(f"Profit Factor: {best['profit_factor']:.2f}")
    print(f"Sharpe Ratio: {best['sharpe_ratio']:.2f}")
    print()
    print("Parameters:")
    for k, v in best['params'].items():
        print(f"  {k}: {v}")

    # Save best config
    output = {
        **best['params'],
        "expected_metrics": {
            "total_return_pct": best['total_return_pct'],
            "win_rate": best['win_rate'],
            "trades_per_day": best['trades_per_day'],
            "max_drawdown_pct": best['max_drawdown_pct'],
            "profit_factor": best['profit_factor'],
        },
        "fee_assumptions": {
            "maker_fee": 0.0,
            "taker_fee": 0.0,
            "note": "Lighter 0% fees"
        }
    }

    output_path = "config/optimized_params_eth_winrate.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {output_path}")

    # Also show some high win-rate configs even if lower ROI
    print()
    print("=" * 80)
    print("HIGHEST WIN RATE CONFIGURATIONS (for reference)")
    print("=" * 80)

    by_winrate = sorted(results, key=lambda x: (x["win_rate"], x["total_return_pct"]), reverse=True)
    print(f"{'#':<3} {'WinRate':<10} {'ROI%':<10} {'Trades/Day':<12} {'MaxDD%':<10} {'PF':<8}")
    print("-" * 80)

    for i, r in enumerate(by_winrate[:10]):
        print(f"{i+1:<3} {r['win_rate']:<10.1f} {r['total_return_pct']:<10.2f} {r['trades_per_day']:<12.1f} {r['max_drawdown_pct']:<10.2f} {r['profit_factor']:<8.2f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/ETHUSDT_1m.csv")
    parser.add_argument("--min-trades", type=float, default=5.0)
    args = parser.parse_args()

    optimize_for_winrate(args.data, args.min_trades)
