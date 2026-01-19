#!/usr/bin/env python3
"""
Optimize RSI+BB Strategy Parameters

Find profitable parameter combinations.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, List
import os
from itertools import product

@dataclass
class Trade:
    entry_price: float
    size: float
    stop_loss: float
    best_price: float
    trailing_active: bool = False

def calculate_rsi(prices: pd.Series, period: int = 7) -> pd.Series:
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_bollinger(prices: pd.Series, period: int = 20, std: float = 2.0):
    sma = prices.rolling(window=period).mean()
    std_dev = prices.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return sma, upper, lower

def backtest(df: pd.DataFrame, rsi_threshold: int, rsi_period: int,
             bb_period: int, bb_std: float, stop_loss_pct: float,
             trailing_pct: float, trailing_activation: float,
             margin: float = 20.0, leverage: float = 20.0,
             fee_rate: float = 0.019/100):
    """Backtest with given parameters."""

    df = df.copy()
    df['rsi'] = calculate_rsi(df['close'], rsi_period)
    df['bb_mid'], df['bb_upper'], df['bb_lower'] = calculate_bollinger(df['close'], bb_period, bb_std)

    notional = margin * leverage
    trades = []
    active_trade = None
    wins = 0
    total_pnl = 0

    for i in range(max(bb_period, rsi_period) + 1, len(df)):
        price = df['close'].iloc[i]
        bb_upper = df['bb_upper'].iloc[i]
        rsi = df['rsi'].iloc[i]

        if active_trade:
            if price < active_trade.best_price:
                active_trade.best_price = price

            profit_pct = (active_trade.entry_price - price) / active_trade.entry_price * 100

            if profit_pct >= trailing_activation and not active_trade.trailing_active:
                active_trade.trailing_active = True

            exit_reason = None

            # Stop loss
            if price >= active_trade.stop_loss:
                exit_reason = "sl"
            # Trailing stop
            elif active_trade.trailing_active:
                trail_price = active_trade.best_price * (1 + trailing_pct / 100)
                if price >= trail_price:
                    exit_reason = "trail"

            if exit_reason:
                pnl = (active_trade.entry_price - price) * active_trade.size
                fees = notional * fee_rate * 2
                pnl -= fees
                total_pnl += pnl
                if pnl > 0:
                    wins += 1
                trades.append(pnl)
                active_trade = None

        else:
            if price > bb_upper and rsi > rsi_threshold:
                size = notional / price
                active_trade = Trade(
                    entry_price=price,
                    size=size,
                    stop_loss=price * (1 + stop_loss_pct / 100),
                    best_price=price,
                )

    win_rate = wins / len(trades) * 100 if trades else 0
    return len(trades), win_rate, total_pnl

def main():
    data_dir = os.path.join(os.path.dirname(__file__), "data")

    # Load BTC data (fastest to test)
    df = pd.read_csv(os.path.join(data_dir, "BTCUSDT_1m.csv"))

    print("=" * 70)
    print("RSI+BB Parameter Optimization - BTC")
    print("=" * 70)
    print("Testing parameter combinations...")
    print()

    # Parameters to test
    rsi_thresholds = [75, 80, 85]
    rsi_periods = [7, 14]
    bb_periods = [20, 30]
    bb_stds = [2.0, 2.5, 3.0]
    stop_losses = [0.30, 0.50, 0.75]
    trailing_pcts = [0.20, 0.30, 0.40]
    trailing_activations = [0.10, 0.15, 0.20]

    results = []

    total_combos = (len(rsi_thresholds) * len(rsi_periods) * len(bb_periods) *
                    len(bb_stds) * len(stop_losses) * len(trailing_pcts) * len(trailing_activations))
    print(f"Testing {total_combos} combinations...")
    print()

    for rsi_thresh, rsi_per, bb_per, bb_std, sl, trail, trail_act in product(
        rsi_thresholds, rsi_periods, bb_periods, bb_stds, stop_losses, trailing_pcts, trailing_activations
    ):
        trades, wr, pnl = backtest(df, rsi_thresh, rsi_per, bb_per, bb_std, sl, trail, trail_act)
        if trades >= 50:  # Minimum trades for significance
            results.append({
                'rsi_thresh': rsi_thresh,
                'rsi_per': rsi_per,
                'bb_per': bb_per,
                'bb_std': bb_std,
                'sl': sl,
                'trail': trail,
                'trail_act': trail_act,
                'trades': trades,
                'wr': wr,
                'pnl': pnl
            })

    # Sort by PnL
    results.sort(key=lambda x: x['pnl'], reverse=True)

    print("TOP 10 BEST PARAMETERS:")
    print("-" * 70)
    print(f"{'RSI_T':<6} {'RSI_P':<6} {'BB_P':<5} {'BB_S':<5} {'SL%':<5} {'TR%':<5} {'ACT%':<5} {'#':<6} {'WR%':<6} {'PnL':<10}")
    print("-" * 70)

    for r in results[:10]:
        print(f"{r['rsi_thresh']:<6} {r['rsi_per']:<6} {r['bb_per']:<5} {r['bb_std']:<5} "
              f"{r['sl']:<5} {r['trail']:<5} {r['trail_act']:<5} "
              f"{r['trades']:<6} {r['wr']:.1f}%{'':<2} ${r['pnl']:.2f}")

    print()
    print("WORST 5 PARAMETERS:")
    print("-" * 70)
    for r in results[-5:]:
        print(f"{r['rsi_thresh']:<6} {r['rsi_per']:<6} {r['bb_per']:<5} {r['bb_std']:<5} "
              f"{r['sl']:<5} {r['trail']:<5} {r['trail_act']:<5} "
              f"{r['trades']:<6} {r['wr']:.1f}%{'':<2} ${r['pnl']:.2f}")

    # Check if any are profitable
    profitable = [r for r in results if r['pnl'] > 0]
    print()
    print(f"Profitable combinations: {len(profitable)} / {len(results)}")

    if profitable:
        best = profitable[0]
        print()
        print("=" * 70)
        print("BEST PROFITABLE PARAMETERS:")
        print("=" * 70)
        print(f"  RSI Threshold: {best['rsi_thresh']}")
        print(f"  RSI Period: {best['rsi_per']}")
        print(f"  BB Period: {best['bb_per']}")
        print(f"  BB Std Dev: {best['bb_std']}")
        print(f"  Stop Loss: {best['sl']}%")
        print(f"  Trailing Stop: {best['trail']}%")
        print(f"  Trailing Activation: {best['trail_act']}%")
        print(f"  Trades: {best['trades']}")
        print(f"  Win Rate: {best['wr']:.1f}%")
        print(f"  Total PnL: ${best['pnl']:.2f}")
    else:
        print()
        print("NO PROFITABLE PARAMETER COMBINATIONS FOUND")
        print("The RSI+BB SHORT strategy may not be viable with current market data.")

if __name__ == "__main__":
    main()
