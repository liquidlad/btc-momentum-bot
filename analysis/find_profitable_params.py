"""
Find Profitable Parameters for High-Volume Trading

Goal: ~50 trades/day, profitable after 0.019% taker fees
"""

import pandas as pd
import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from strategy.backtest import Backtester

# Fee constant
FEE_RATE = 0.00019  # 0.019% per side

def run_backtest_with_fees(df, params, capital=200):
    """Run backtest and calculate net P&L after fees."""
    bt = Backtester(params, capital=capital, leverage=50)
    results = bt.run(df)

    # Calculate total trading volume and fees
    total_volume = sum(trade.size * 2 for trade in bt.trades)  # entry + exit
    fee_cost = total_volume * FEE_RATE

    # Net results
    gross_pnl = results['total_pnl']
    net_pnl = gross_pnl - fee_cost
    trading_days = 90

    return {
        'total_trades': results['total_trades'],
        'trades_per_day': results['trades_per_day'],
        'win_rate': results['win_rate'],
        'gross_pnl': gross_pnl,
        'fee_cost': fee_cost,
        'net_pnl': net_pnl,
        'net_return_pct': net_pnl / capital * 100,
        'daily_gross': gross_pnl / trading_days,
        'daily_fees': fee_cost / trading_days,
        'daily_net': net_pnl / trading_days,
        'max_drawdown': results['max_drawdown_pct'],
        'profit_factor': results['profit_factor'],
        'avg_trade_gross': gross_pnl / results['total_trades'] if results['total_trades'] > 0 else 0,
        'avg_trade_fee': fee_cost / results['total_trades'] if results['total_trades'] > 0 else 0,
    }

def main():
    # Load data
    print("Loading BTC 1m data...")
    df = pd.read_csv('data/BTCUSDT_1m.csv', parse_dates=['open_time'], index_col='open_time')
    print(f"Loaded {len(df)} candles\n")

    # Base params that don't change
    base_params = {
        'roc_period': 3,
        'volume_sma_period': 20,
        'rsi_period': 14,
        'rsi_long_min': 25,
        'rsi_long_max': 75,
        'rsi_short_min': 25,
        'rsi_short_max': 75,
        'take_profit_2_pct': 0.30,  # Will vary
        'risk_per_trade_pct': 1.0,
        'max_position_notional': 10000,
        'max_hold_candles': 8,
        'max_candles_in_trade': 8,
        'trailing_stop_pct': 0.05,
        'trailing_activation_pct': 0.08,
    }

    # Parameter combinations to test
    # Focus on wider targets since we know tight targets don't work with fees
    test_configs = [
        # (min_cond, ema_fast, ema_slow, roc_thresh, vol_mult, sl%, tp%)
        # Current baseline
        (3, 3, 15, 0.08, 1.0, 0.10, 0.12),

        # Wider targets - same entry conditions
        (3, 3, 15, 0.08, 1.0, 0.15, 0.20),
        (3, 3, 15, 0.08, 1.0, 0.15, 0.25),
        (3, 3, 15, 0.08, 1.0, 0.15, 0.30),
        (3, 3, 15, 0.08, 1.0, 0.20, 0.30),
        (3, 3, 15, 0.08, 1.0, 0.20, 0.40),
        (3, 3, 15, 0.08, 1.0, 0.25, 0.50),

        # Very wide targets
        (3, 3, 15, 0.08, 1.0, 0.30, 0.60),
        (3, 3, 15, 0.08, 1.0, 0.40, 0.80),
        (3, 3, 15, 0.08, 1.0, 0.50, 1.00),

        # Adjust entry conditions with wide targets
        (2, 3, 15, 0.05, 1.0, 0.20, 0.40),  # More trades
        (2, 3, 10, 0.05, 1.0, 0.20, 0.40),  # Faster EMA
        (2, 5, 20, 0.05, 1.0, 0.25, 0.50),  # Slower EMA

        # Higher conviction with wide targets
        (4, 3, 15, 0.08, 1.0, 0.20, 0.40),
        (4, 3, 15, 0.08, 1.0, 0.25, 0.50),
        (4, 3, 15, 0.10, 1.2, 0.30, 0.60),

        # Asymmetric risk/reward (wide TP, tighter SL)
        (3, 3, 15, 0.08, 1.0, 0.12, 0.30),
        (3, 3, 15, 0.08, 1.0, 0.15, 0.40),
        (3, 3, 15, 0.08, 1.0, 0.15, 0.50),
        (3, 3, 15, 0.08, 1.0, 0.20, 0.60),

        # More aggressive asymmetry
        (3, 3, 15, 0.08, 1.0, 0.10, 0.40),
        (3, 3, 15, 0.08, 1.0, 0.10, 0.50),
        (3, 3, 15, 0.08, 1.0, 0.12, 0.50),

        # Looser entry + very wide targets
        (2, 3, 15, 0.05, 0.8, 0.20, 0.50),
        (2, 3, 15, 0.05, 0.8, 0.25, 0.60),
        (2, 5, 15, 0.06, 1.0, 0.20, 0.50),
    ]

    results = []

    print("=" * 100)
    print("PARAMETER SEARCH: Finding profitable setups with ~50 trades/day")
    print("=" * 100)
    print(f"{'Config':<40} {'Trades/Day':>10} {'WinRate':>8} {'Gross':>10} {'Fees':>10} {'NET':>10} {'Status':<10}")
    print("-" * 100)

    for config in test_configs:
        min_cond, ema_fast, ema_slow, roc_thresh, vol_mult, sl_pct, tp_pct = config

        params = base_params.copy()
        params['min_conditions'] = min_cond
        params['ema_fast'] = ema_fast
        params['ema_slow'] = ema_slow
        params['roc_threshold'] = roc_thresh
        params['volume_multiplier'] = vol_mult
        params['stop_loss_pct'] = sl_pct
        params['take_profit_1_pct'] = tp_pct
        params['take_profit_2_pct'] = tp_pct * 1.5  # Second TP at 1.5x first

        try:
            result = run_backtest_with_fees(df, params)
            result['config'] = config
            results.append(result)

            config_str = f"c{min_cond} e{ema_fast}/{ema_slow} r{roc_thresh} v{vol_mult} sl{sl_pct} tp{tp_pct}"
            status = "PROFIT" if result['net_pnl'] > 0 else "LOSS"

            print(f"{config_str:<40} {result['trades_per_day']:>10.1f} {result['win_rate']:>7.1f}% "
                  f"${result['gross_pnl']:>9.0f} ${result['fee_cost']:>9.0f} ${result['net_pnl']:>9.0f} {status:<10}")
        except Exception as e:
            print(f"{str(config):<40} ERROR: {e}")

    print("\n" + "=" * 100)
    print("TOP RESULTS (sorted by Net P&L)")
    print("=" * 100)

    # Sort by net P&L
    results.sort(key=lambda x: x['net_pnl'], reverse=True)

    print(f"\n{'Rank':<5} {'Config':<45} {'Trades/Day':>10} {'WinRate':>8} {'Net P&L':>12} {'Daily Net':>10}")
    print("-" * 100)

    for i, r in enumerate(results[:15]):
        config = r['config']
        min_cond, ema_fast, ema_slow, roc_thresh, vol_mult, sl_pct, tp_pct = config
        config_str = f"min_cond={min_cond} EMA={ema_fast}/{ema_slow} SL={sl_pct}% TP={tp_pct}%"

        print(f"{i+1:<5} {config_str:<45} {r['trades_per_day']:>10.1f} {r['win_rate']:>7.1f}% "
              f"${r['net_pnl']:>11.2f} ${r['daily_net']:>9.2f}")

    # Find best result with ~50 trades/day
    print("\n" + "=" * 100)
    print("BEST RESULTS WITH 30-70 TRADES/DAY (target volume range)")
    print("=" * 100)

    volume_filtered = [r for r in results if 30 <= r['trades_per_day'] <= 70]
    volume_filtered.sort(key=lambda x: x['net_pnl'], reverse=True)

    if volume_filtered:
        print(f"\n{'Rank':<5} {'Config':<45} {'Trades/Day':>10} {'WinRate':>8} {'Net P&L':>12} {'Daily Net':>10}")
        print("-" * 100)

        for i, r in enumerate(volume_filtered[:10]):
            config = r['config']
            min_cond, ema_fast, ema_slow, roc_thresh, vol_mult, sl_pct, tp_pct = config
            config_str = f"min_cond={min_cond} EMA={ema_fast}/{ema_slow} SL={sl_pct}% TP={tp_pct}%"

            print(f"{i+1:<5} {config_str:<45} {r['trades_per_day']:>10.1f} {r['win_rate']:>7.1f}% "
                  f"${r['net_pnl']:>11.2f} ${r['daily_net']:>9.2f}")

        best = volume_filtered[0]
        print(f"\n*** BEST CONFIG FOR ~50 TRADES/DAY ***")
        print(f"Parameters: {best['config']}")
        print(f"Trades/Day: {best['trades_per_day']:.1f}")
        print(f"Win Rate: {best['win_rate']:.1f}%")
        print(f"Gross P&L: ${best['gross_pnl']:.2f}")
        print(f"Fee Cost: ${best['fee_cost']:.2f}")
        print(f"Net P&L: ${best['net_pnl']:.2f}")
        print(f"Daily Net: ${best['daily_net']:.2f}")
        print(f"Max Drawdown: {best['max_drawdown']:.1f}%")
    else:
        print("No configurations found in target volume range. Try adjusting parameters.")

    # Profitable configs only
    profitable = [r for r in results if r['net_pnl'] > 0]
    print(f"\n\nTotal profitable configs: {len(profitable)} / {len(results)}")

    if profitable:
        print("\nALL PROFITABLE CONFIGURATIONS:")
        for r in profitable:
            config = r['config']
            min_cond, ema_fast, ema_slow, roc_thresh, vol_mult, sl_pct, tp_pct = config
            print(f"  SL={sl_pct}% TP={tp_pct}% min_cond={min_cond}: "
                  f"{r['trades_per_day']:.0f} trades/day, ${r['daily_net']:.2f}/day net")

if __name__ == "__main__":
    main()
