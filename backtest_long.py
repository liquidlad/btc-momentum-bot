#!/usr/bin/env python3
"""Test LONG strategy: Buy when RSI < threshold and price < lower BB."""

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

def backtest_long(df, rsi_thresh, sl_pct, tp_pct, fee=0.00038):
    """LONG when RSI < thresh and price < lower BB."""
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
        bb_lower = df['bb_lower'].iloc[i]

        if in_trade:
            pnl_pct = (price - entry_price) / entry_price * 100  # LONG: profit when price goes UP
            if pnl_pct <= -sl_pct:
                pnl = -sl_pct / 100 * notional - (notional * fee)
                trades.append(('sl', pnl))
                in_trade = False
            elif pnl_pct >= tp_pct:
                pnl = tp_pct / 100 * notional - (notional * fee)
                trades.append(('tp', pnl))
                in_trade = False
        else:
            # LONG entry: price < lower BB and RSI < threshold (oversold)
            if price < bb_lower and rsi < rsi_thresh:
                entry_price = price
                in_trade = True

    if trades:
        total_pnl = sum(t[1] for t in trades)
        wins = sum(1 for t in trades if t[1] > 0)
        return len(trades), wins/len(trades)*100, total_pnl
    return 0, 0, 0

# Load data
data_dir = os.path.join(os.path.dirname(__file__), "data")
df_btc = pd.read_csv(os.path.join(data_dir, "BTCUSDT_1m.csv"))
df_eth = pd.read_csv(os.path.join(data_dir, "ETHUSDT_1m.csv"))
df_sol = pd.read_csv(os.path.join(data_dir, "SOLUSDT_1m.csv"))

print("=" * 70)
print("RSI+BB LONG Strategy - Buy Oversold")
print("=" * 70)
print("Entry: LONG when price < lower BB AND RSI < threshold")
print("Fixed: BB(20,2), RSI(7), Margin $20 x 20x, Fee 0.038% round-trip")
print()

params = [
    # (RSI threshold, Stop Loss %, Take Profit %)
    (25, 0.30, 0.20),
    (25, 0.50, 0.30),
    (25, 0.50, 0.50),
    (30, 0.30, 0.20),
    (30, 0.50, 0.30),
    (30, 0.50, 0.50),
    (35, 0.50, 0.30),
    (35, 0.50, 0.50),
    (20, 0.50, 0.50),
    (20, 0.75, 0.50),
    (15, 0.50, 0.50),
]

print(f"{'RSI<':<5} {'SL%':<6} {'TP%':<6} {'BTC#':<8} {'BTC PnL':<12} {'ETH PnL':<12} {'SOL PnL':<12} {'TOTAL':<12}")
print("-" * 85)

best_pnl = -999999
best_params = None

for rsi_t, sl, tp in params:
    btc_n, btc_wr, btc_pnl = backtest_long(df_btc, rsi_t, sl, tp)
    eth_n, eth_wr, eth_pnl = backtest_long(df_eth, rsi_t + 15, sl, tp)  # ETH more trades
    sol_n, sol_wr, sol_pnl = backtest_long(df_sol, rsi_t + 10, sl, tp)
    total = btc_pnl + eth_pnl + sol_pnl

    print(f"{rsi_t:<5} {sl:<6} {tp:<6} {btc_n:<8} ${btc_pnl:<11.2f} ${eth_pnl:<11.2f} ${sol_pnl:<11.2f} ${total:<11.2f}")

    if total > best_pnl:
        best_pnl = total
        best_params = (rsi_t, sl, tp)

print()
print("=" * 70)
if best_pnl > 0:
    print(f"PROFITABLE! RSI<{best_params[0]}, SL={best_params[1]}%, TP={best_params[2]}% -> ${best_pnl:.2f}")
else:
    print(f"BEST: RSI<{best_params[0]}, SL={best_params[1]}%, TP={best_params[2]}% -> ${best_pnl:.2f}")
print("=" * 70)
