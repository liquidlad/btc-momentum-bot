"""
Backtest BB Short (Fixed SL/TP) for comparison with RSI+BB
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import argparse


def calculate_bb(prices: np.ndarray, period: int = 20, std_mult: float = 2.0):
    """Calculate Bollinger Bands."""
    prices_series = pd.Series(prices)
    middle = prices_series.rolling(window=period, min_periods=period).mean().values
    std = prices_series.rolling(window=period, min_periods=period).std().values
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    return lower, middle, upper


def run_backtest(df: pd.DataFrame, sl_pct: float = 0.3, tp_pct: float = 0.4) -> Dict:
    """Run BB Short backtest with fixed SL/TP."""
    prices = df['close'].values
    n = len(prices)

    lower_bb, middle_bb, upper_bb = calculate_bb(prices, 20, 2.0)

    trades = []
    in_trade = False
    entry_price = 0
    entry_idx = 0

    for i in range(21, n):
        if in_trade:
            # Check SL (price up = loss for short)
            sl_price = entry_price * (1 + sl_pct / 100)
            tp_price = entry_price * (1 - tp_pct / 100)

            if prices[i] >= sl_price:
                pnl_pct = -sl_pct
                trades.append({'pnl_pct': pnl_pct, 'exit_reason': 'stop_loss', 'hold_bars': i - entry_idx})
                in_trade = False
            elif prices[i] <= tp_price:
                pnl_pct = tp_pct
                trades.append({'pnl_pct': pnl_pct, 'exit_reason': 'take_profit', 'hold_bars': i - entry_idx})
                in_trade = False
        else:
            # Entry: price > upper BB
            if not np.isnan(upper_bb[i]) and prices[i] > upper_bb[i]:
                in_trade = True
                entry_price = prices[i]
                entry_idx = i

    # Close open trade
    if in_trade:
        pnl_pct = (entry_price - prices[-1]) / entry_price * 100
        trades.append({'pnl_pct': pnl_pct, 'exit_reason': 'end_of_data', 'hold_bars': n - 1 - entry_idx})

    return analyze(trades, df)


def analyze(trades: List[Dict], df: pd.DataFrame) -> Dict:
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

    # Exit reasons
    exit_reasons = {}
    for t in trades:
        r = t['exit_reason']
        exit_reasons[r] = exit_reasons.get(r, 0) + 1

    avg_hold = np.mean([t['hold_bars'] for t in trades])

    return {
        "total_trades": len(trades),
        "trades_per_day": len(trades) / trading_days,
        "win_rate": win_rate,
        "total_pnl_usd": total_pnl_usd,
        "max_drawdown_pct": max_dd,
        "avg_hold_bars": avg_hold,
        "exit_reasons": exit_reasons,
    }


def main():
    assets = [
        ("BTC", "data/BTCUSDT_1m.csv"),
        ("ETH", "data/ETHUSDT_1m.csv"),
        ("SOL", "data/SOLUSDT_1m.csv"),
    ]

    print("=" * 80)
    print("BB SHORT (Fixed SL/TP) vs RSI+BB COMPARISON")
    print("=" * 80)
    print()
    print("BB Short Strategy: Entry when price > upper BB, Exit at SL 0.3% / TP 0.4%")
    print()

    total_pnl = 0
    total_trades = 0

    print(f"{'Asset':<8} {'PnL$':<12} {'WinRate':<10} {'MaxDD%':<10} {'Trades':<10} {'Tr/Day':<10} {'TP Exits':<10} {'SL Exits':<10}")
    print("-" * 80)

    for asset, data_path in assets:
        df = pd.read_csv(data_path, parse_dates=["open_time"], index_col="open_time")
        result = run_backtest(df, sl_pct=0.3, tp_pct=0.4)

        if "error" not in result:
            tp_exits = result['exit_reasons'].get('take_profit', 0)
            sl_exits = result['exit_reasons'].get('stop_loss', 0)

            print(f"{asset:<8} ${result['total_pnl_usd']:<11.0f} {result['win_rate']:<10.1f} {result['max_drawdown_pct']:<10.1f} {result['total_trades']:<10} {result['trades_per_day']:<10.1f} {tp_exits:<10} {sl_exits:<10}")

            total_pnl += result['total_pnl_usd']
            total_trades += result['total_trades']

    print("-" * 80)
    print(f"{'TOTAL':<8} ${total_pnl:<11.0f}")
    print()

    # Now show RSI+BB results for comparison
    print("=" * 80)
    print("RSI+BB (Trailing Stop) - From previous backtest")
    print("=" * 80)
    print()
    print(f"{'Asset':<8} {'PnL$':<12} {'WinRate':<10} {'MaxDD%':<10}")
    print("-" * 80)
    print(f"{'BTC':<8} ${'1013':<11} {'56.2':<10} {'40.9':<10}")
    print(f"{'ETH':<8} ${'1371':<11} {'52.4':<10} {'43.8':<10}")
    print(f"{'SOL':<8} ${'1792':<11} {'51.6':<10} {'51.1':<10}")
    print("-" * 80)
    print(f"{'TOTAL':<8} ${'4176':<11}")
    print()

    print("=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    improvement = ((4176 - total_pnl) / total_pnl * 100) if total_pnl > 0 else 0
    print(f"BB Short (SL/TP):     ${total_pnl:.0f}")
    print(f"RSI+BB (Trailing):    $4,176")
    print(f"Improvement:          {improvement:+.1f}%")


if __name__ == "__main__":
    main()
