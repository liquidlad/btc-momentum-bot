#!/usr/bin/env python3
"""Test strategies with 0% fees (Lighter exchange)."""

import pandas as pd
import numpy as np
import os

def calculate_rsi(prices, period=7):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_bb(prices, period=20, std=2.0):
    sma = prices.rolling(window=period).mean()
    std_dev = prices.rolling(window=period).std()
    return sma + (std * std_dev), sma - (std * std_dev)

def backtest(df, direction, rsi_thresh, sl_pct, tp_pct, fee=0):
    """Backtest with variable fee."""
    df = df.copy()
    df['rsi'] = calculate_rsi(df['close'], 7)
    df['bb_upper'], df['bb_lower'] = calculate_bb(df['close'], 20, 2.0)

    trades = []
    in_trade = False
    entry_price = 0
    notional = 400

    for i in range(21, len(df)):
        price = df['close'].iloc[i]
        rsi = df['rsi'].iloc[i]
        bb_upper = df['bb_upper'].iloc[i]
        bb_lower = df['bb_lower'].iloc[i]

        if in_trade:
            if direction == "SHORT":
                pnl_pct = (entry_price - price) / entry_price * 100
            else:
                pnl_pct = (price - entry_price) / entry_price * 100

            if pnl_pct <= -sl_pct:
                pnl = -sl_pct / 100 * notional - (notional * fee)
                trades.append(pnl)
                in_trade = False
            elif pnl_pct >= tp_pct:
                pnl = tp_pct / 100 * notional - (notional * fee)
                trades.append(pnl)
                in_trade = False
        else:
            if direction == "SHORT":
                if price > bb_upper and rsi > rsi_thresh:
                    entry_price = price
                    in_trade = True
            else:
                if price < bb_lower and rsi < rsi_thresh:
                    entry_price = price
                    in_trade = True

    if trades:
        return len(trades), sum(1 for t in trades if t > 0)/len(trades)*100, sum(trades)
    return 0, 0, 0

# Load data
data_dir = os.path.join(os.path.dirname(__file__), "data")
df_btc = pd.read_csv(os.path.join(data_dir, "BTCUSDT_1m.csv"))

print("=" * 70)
print("Fee Impact Analysis - BTC Only")
print("=" * 70)
print()

# Test SHORT strategy
print("SHORT Strategy (RSI > 80, SL 0.5%, TP 0.5%)")
print("-" * 50)
for fee_name, fee in [("Paradex 0.038%", 0.00038), ("Lighter 0%", 0)]:
    n, wr, pnl = backtest(df_btc, "SHORT", 80, 0.5, 0.5, fee)
    print(f"{fee_name:<20}: {n} trades, WR: {wr:.1f}%, PnL: ${pnl:.2f}")

print()
print("LONG Strategy (RSI < 20, SL 0.5%, TP 0.5%)")
print("-" * 50)
for fee_name, fee in [("Paradex 0.038%", 0.00038), ("Lighter 0%", 0)]:
    n, wr, pnl = backtest(df_btc, "LONG", 20, 0.5, 0.5, fee)
    print(f"{fee_name:<20}: {n} trades, WR: {wr:.1f}%, PnL: ${pnl:.2f}")

print()
print("=" * 70)
print("Testing different params with 0% fee:")
print("=" * 70)
print()

best_pnl = -99999
best_config = None

for direction in ["SHORT", "LONG"]:
    for rsi in [75, 80, 85, 90] if direction == "SHORT" else [15, 20, 25, 30]:
        for sl in [0.3, 0.5, 0.75]:
            for tp in [0.3, 0.5, 0.75]:
                n, wr, pnl = backtest(df_btc, direction, rsi, sl, tp, 0)
                if n >= 50 and pnl > best_pnl:
                    best_pnl = pnl
                    best_config = (direction, rsi, sl, tp, n, wr)

if best_config:
    d, rsi, sl, tp, n, wr = best_config
    print(f"BEST CONFIG (0% fee): {d} RSI {'>' if d=='SHORT' else '<'}{rsi}, SL={sl}%, TP={tp}%")
    print(f"  Trades: {n}, Win Rate: {wr:.1f}%, PnL: ${best_pnl:.2f}")
else:
    print("No profitable configuration found even with 0% fees!")
