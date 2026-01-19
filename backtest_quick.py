#!/usr/bin/env python3
"""Quick backtest with fewer combinations."""

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

def backtest(df, rsi_thresh, sl_pct, tp_pct, fee=0.00038):
    """Simple backtest: entry when RSI > thresh and price > upper BB, fixed SL/TP."""
    df = df.copy()
    df['rsi'] = calculate_rsi(df['close'], 7)
    df['bb_upper'], df['bb_lower'] = calculate_bb(df['close'], 20, 2.0)

    trades = []
    in_trade = False
    entry_price = 0
    notional = 400  # $20 margin x 20 leverage

    for i in range(21, len(df)):
        price = df['close'].iloc[i]
        rsi = df['rsi'].iloc[i]
        bb_upper = df['bb_upper'].iloc[i]

        if in_trade:
            # Check SL/TP
            pnl_pct = (entry_price - price) / entry_price * 100
            if pnl_pct <= -sl_pct:  # Stop loss hit
                pnl = -sl_pct / 100 * notional - (notional * fee)
                trades.append(('sl', pnl))
                in_trade = False
            elif pnl_pct >= tp_pct:  # Take profit hit
                pnl = tp_pct / 100 * notional - (notional * fee)
                trades.append(('tp', pnl))
                in_trade = False
        else:
            if price > bb_upper and rsi > rsi_thresh:
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
print("RSI+BB SHORT Strategy - Parameter Search")
print("=" * 70)
print("Fixed: BB(20,2), RSI(7), Margin $20 x 20x, Fee 0.038% round-trip")
print()

# Test different RSI thresholds and SL/TP combinations
params = [
    # (RSI threshold, Stop Loss %, Take Profit %)
    (75, 0.30, 0.20),  # Current
    (75, 0.50, 0.30),
    (75, 0.50, 0.50),
    (80, 0.30, 0.20),
    (80, 0.50, 0.30),
    (80, 0.50, 0.50),
    (85, 0.50, 0.30),
    (85, 0.50, 0.50),
    (85, 0.75, 0.50),
    (90, 0.50, 0.50),
    (90, 0.75, 0.50),
]

print(f"{'RSI':<5} {'SL%':<6} {'TP%':<6} {'BTC Trades':<12} {'BTC PnL':<12} {'ETH PnL':<12} {'SOL PnL':<12} {'TOTAL':<12}")
print("-" * 85)

best_pnl = -999999
best_params = None

for rsi_t, sl, tp in params:
    btc_n, btc_wr, btc_pnl = backtest(df_btc, rsi_t, sl, tp)
    eth_n, eth_wr, eth_pnl = backtest(df_eth, rsi_t - 15, sl, tp)  # ETH uses RSI-15
    sol_n, sol_wr, sol_pnl = backtest(df_sol, rsi_t - 10, sl, tp)  # SOL uses RSI-10
    total = btc_pnl + eth_pnl + sol_pnl

    print(f"{rsi_t:<5} {sl:<6} {tp:<6} {btc_n:<12} ${btc_pnl:<11.2f} ${eth_pnl:<11.2f} ${sol_pnl:<11.2f} ${total:<11.2f}")

    if total > best_pnl:
        best_pnl = total
        best_params = (rsi_t, sl, tp)

print()
print("=" * 70)
print(f"BEST: RSI>{best_params[0]}, SL={best_params[1]}%, TP={best_params[2]}% -> ${best_pnl:.2f}")
print("=" * 70)

if best_pnl < 0:
    print()
    print("*** ALL COMBINATIONS LOSE MONEY ***")
    print("The RSI+BB SHORT strategy is not profitable with these parameters.")
    print()
    print("RECOMMENDATION: Try LONG entries when RSI < threshold and price < lower BB")
