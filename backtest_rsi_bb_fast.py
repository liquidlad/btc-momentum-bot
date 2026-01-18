"""
RSI + Bollinger Band Combo Strategy Backtest (FAST VERSION)

Uses vectorized operations for speed.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import argparse
import itertools


def calculate_rsi_vectorized(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate RSI using vectorized operations."""
    deltas = np.diff(prices, prepend=prices[0])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    # Use pandas for rolling mean (faster than manual loop)
    gains_series = pd.Series(gains)
    losses_series = pd.Series(losses)

    avg_gains = gains_series.rolling(window=period, min_periods=period).mean().values
    avg_losses = losses_series.rolling(window=period, min_periods=period).mean().values

    rs = np.divide(avg_gains, avg_losses, out=np.ones_like(avg_gains), where=avg_losses != 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.nan_to_num(rsi, nan=50)

    return rsi


def calculate_bb_vectorized(prices: np.ndarray, period: int = 20, std_mult: float = 2.0):
    """Calculate Bollinger Bands using vectorized operations."""
    prices_series = pd.Series(prices)

    middle = prices_series.rolling(window=period, min_periods=period).mean().values
    std = prices_series.rolling(window=period, min_periods=period).std().values

    upper = middle + std_mult * std
    lower = middle - std_mult * std

    return lower, middle, upper


def run_backtest_fast(df: pd.DataFrame, params: Dict) -> Dict:
    """Run backtest using vectorized operations where possible."""
    bb_period = params['bb_period']
    bb_std = params['bb_std']
    rsi_period = params['rsi_period']
    rsi_entry = params['rsi_entry']
    rsi_exit = params['rsi_exit']
    exit_mode = params['exit_mode']
    stop_loss_pct = params.get('stop_loss_pct')

    prices = df['close'].values
    n = len(prices)

    # Pre-calculate indicators
    rsi = calculate_rsi_vectorized(prices, rsi_period)
    lower_bb, middle_bb, upper_bb = calculate_bb_vectorized(prices, bb_period, bb_std)

    # Simulate trades
    trades = []
    in_trade = False
    entry_price = 0
    entry_idx = 0

    start_idx = max(bb_period, rsi_period) + 1

    for i in range(start_idx, n):
        if in_trade:
            # Check exit
            should_exit = False
            exit_reason = None

            # Stop loss
            if stop_loss_pct and prices[i] >= entry_price * (1 + stop_loss_pct / 100):
                should_exit = True
                exit_reason = "stop_loss"

            # Exit conditions
            if not should_exit:
                if exit_mode == 'rsi' and rsi[i] < rsi_exit:
                    should_exit = True
                    exit_reason = "rsi_exit"
                elif exit_mode == 'lower_bb' and prices[i] < lower_bb[i]:
                    should_exit = True
                    exit_reason = "lower_bb"
                elif exit_mode == 'rsi_or_bb':
                    if rsi[i] < rsi_exit:
                        should_exit = True
                        exit_reason = "rsi_exit"
                    elif prices[i] < lower_bb[i]:
                        should_exit = True
                        exit_reason = "lower_bb"

            if should_exit:
                pnl_pct = (entry_price - prices[i]) / entry_price * 100
                trades.append({
                    'pnl_pct': pnl_pct,
                    'exit_reason': exit_reason,
                    'hold_bars': i - entry_idx
                })
                in_trade = False

        else:
            # Check entry: price > upper BB AND RSI > threshold
            if not np.isnan(upper_bb[i]) and prices[i] > upper_bb[i] and rsi[i] > rsi_entry:
                in_trade = True
                entry_price = prices[i]
                entry_idx = i

    # Close open trade
    if in_trade:
        pnl_pct = (entry_price - prices[-1]) / entry_price * 100
        trades.append({
            'pnl_pct': pnl_pct,
            'exit_reason': 'end_of_data',
            'hold_bars': n - 1 - entry_idx
        })

    return analyze_trades_fast(trades, df)


def analyze_trades_fast(trades: List[Dict], df: pd.DataFrame) -> Dict:
    """Analyze trades."""
    trading_days = (df.index[-1] - df.index[0]).days or 1

    if not trades:
        return {"error": "No trades", "total_trades": 0}

    leverage = 20.0
    margin = 100.0
    notional = margin * leverage

    pnls = [t['pnl_pct'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_pnl_pct = sum(pnls)
    total_pnl_usd = total_pnl_pct / 100 * notional

    win_rate = len(wins) / len(trades) * 100

    # Exit reasons
    exit_reasons = {}
    for t in trades:
        r = t['exit_reason']
        exit_reasons[r] = exit_reasons.get(r, 0) + 1

    # Max drawdown
    equity = [margin]
    for p in pnls:
        equity.append(equity[-1] + p / 100 * notional)

    peak = equity[0]
    max_dd = 0
    for e in equity:
        if e > peak:
            peak = e
        dd = (peak - e) / peak * 100 if peak > 0 else 0
        max_dd = max(max_dd, dd)

    # Profit factor
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.001
    profit_factor = gross_profit / gross_loss

    avg_hold = np.mean([t['hold_bars'] for t in trades])

    return {
        "total_trades": len(trades),
        "trades_per_day": len(trades) / trading_days,
        "win_rate": win_rate,
        "total_pnl_pct": total_pnl_pct,
        "total_pnl_usd": total_pnl_usd,
        "max_drawdown_pct": max_dd,
        "avg_hold_bars": avg_hold,
        "profit_factor": profit_factor,
        "exit_reasons": exit_reasons,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/BTCUSDT_1m.csv")
    parser.add_argument("--asset", default="BTC")
    args = parser.parse_args()

    print(f"Loading {args.data}...")
    df = pd.read_csv(args.data, parse_dates=["open_time"], index_col="open_time")
    trading_days = (df.index[-1] - df.index[0]).days or 1
    print(f"Data: {len(df)} candles over {trading_days} days")
    print()

    # Parameter combinations
    param_grid = {
        'bb_period': [20],
        'bb_std': [2.0],
        'rsi_period': [7, 14],
        'rsi_entry': [60, 65, 70, 75, 80],
        'rsi_exit': [30, 40, 50, 55],
        'exit_mode': ['rsi', 'lower_bb', 'rsi_or_bb'],
        'stop_loss_pct': [None, 0.2, 0.3, 0.5, 1.0],
    }

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))

    print(f"Testing {len(combinations)} combinations for {args.asset}...")

    results = []
    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        metrics = run_backtest_fast(df, params)

        if metrics.get("total_trades", 0) >= 10 and metrics.get("total_pnl_usd", 0) > 0:
            metrics["params"] = params
            results.append(metrics)

        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(combinations)}", end="\r")

    print(f"\nFound {len(results)} profitable configurations")
    print()

    if not results:
        print("No profitable configs found!")
        return

    # Sort by PnL
    results.sort(key=lambda x: x["total_pnl_usd"], reverse=True)

    print("=" * 130)
    print(f"TOP 15 BY PNL - {args.asset}")
    print("=" * 130)
    print(f"{'#':<3} {'PnL$':<10} {'WR%':<8} {'Tr/Day':<8} {'MaxDD%':<8} {'PF':<6} {'Hold':<6} {'RSI_P':<6} {'Entry':<6} {'Exit':<6} {'Mode':<12} {'SL':<8}")
    print("-" * 130)

    for i, r in enumerate(results[:15]):
        p = r['params']
        sl = f"{p['stop_loss_pct']}%" if p['stop_loss_pct'] else "None"
        print(f"{i+1:<3} ${r['total_pnl_usd']:<9.0f} {r['win_rate']:<8.1f} {r['trades_per_day']:<8.1f} {r['max_drawdown_pct']:<8.1f} {r['profit_factor']:<6.2f} {r['avg_hold_bars']:<6.0f} {p['rsi_period']:<6} {p['rsi_entry']:<6} {p['rsi_exit']:<6} {p['exit_mode']:<12} {sl:<8}")

    # Best by win rate (min 5 trades/day)
    wr_results = [r for r in results if r['trades_per_day'] >= 3]
    if wr_results:
        wr_results.sort(key=lambda x: x['win_rate'], reverse=True)
        print()
        print("=" * 130)
        print(f"TOP 10 BY WIN RATE (min 3 tr/day) - {args.asset}")
        print("=" * 130)
        print(f"{'#':<3} {'WR%':<8} {'PnL$':<10} {'Tr/Day':<8} {'MaxDD%':<8} {'PF':<6} {'RSI_P':<6} {'Entry':<6} {'Exit':<6} {'Mode':<12} {'SL':<8}")
        print("-" * 130)

        for i, r in enumerate(wr_results[:10]):
            p = r['params']
            sl = f"{p['stop_loss_pct']}%" if p['stop_loss_pct'] else "None"
            print(f"{i+1:<3} {r['win_rate']:<8.1f} ${r['total_pnl_usd']:<9.0f} {r['trades_per_day']:<8.1f} {r['max_drawdown_pct']:<8.1f} {r['profit_factor']:<6.2f} {p['rsi_period']:<6} {p['rsi_entry']:<6} {p['rsi_exit']:<6} {p['exit_mode']:<12} {sl:<8}")

    # Best config details
    best = results[0]
    print()
    print("=" * 130)
    print("BEST CONFIG DETAILS")
    print("=" * 130)
    print(f"PnL: ${best['total_pnl_usd']:.2f}")
    print(f"Win Rate: {best['win_rate']:.1f}%")
    print(f"Trades/Day: {best['trades_per_day']:.1f}")
    print(f"Max DD: {best['max_drawdown_pct']:.1f}%")
    print(f"Profit Factor: {best['profit_factor']:.2f}")
    print()
    print("Parameters:")
    for k, v in best['params'].items():
        print(f"  {k}: {v}")
    print()
    print("Exit Reasons:")
    for k, v in best['exit_reasons'].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
