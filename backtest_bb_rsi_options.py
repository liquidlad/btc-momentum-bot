#!/usr/bin/env python3
"""
Backtest RSI+BB Strategy Options

Tests 3 exit strategies:
A) No lower BB exit - just trailing stop + stop loss
B) Lock lower BB at entry - exit only at original target
C) Lower BB exit only if in profit

Entry: SHORT when price > upper BB(20) AND RSI(7) > threshold
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, List
import os

# RSI thresholds per asset
RSI_THRESHOLDS = {"BTC": 75, "ETH": 60, "SOL": 65}

@dataclass
class Trade:
    entry_idx: int
    entry_price: float
    entry_bb_lower: float  # BB lower at entry time
    size: float
    stop_loss: float
    best_price: float
    trailing_active: bool = False

@dataclass
class TradeResult:
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    exit_reason: str
    hold_bars: int

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

def backtest_option(df: pd.DataFrame, asset: str, option: str,
                    margin: float = 20.0, leverage: float = 20.0,
                    stop_loss_pct: float = 0.30, trailing_pct: float = 0.20,
                    trailing_activation: float = 0.10, fee_rate: float = 0.019/100) -> List[TradeResult]:
    """
    Backtest a single option.

    Options:
    A: No lower BB exit
    B: Lock BB at entry
    C: Lower BB only if in profit
    """

    # Calculate indicators
    df = df.copy()
    df['rsi'] = calculate_rsi(df['close'], 7)
    df['bb_mid'], df['bb_upper'], df['bb_lower'] = calculate_bollinger(df['close'], 20, 2.0)

    rsi_threshold = RSI_THRESHOLDS.get(asset, 75)
    notional = margin * leverage

    trades: List[TradeResult] = []
    active_trade: Optional[Trade] = None

    for i in range(20, len(df)):
        price = df['close'].iloc[i]
        bb_upper = df['bb_upper'].iloc[i]
        bb_lower = df['bb_lower'].iloc[i]
        rsi = df['rsi'].iloc[i]

        if active_trade:
            hold_bars = i - active_trade.entry_idx

            # Update best price (lowest for short)
            if price < active_trade.best_price:
                active_trade.best_price = price

            # Check profit
            profit_pct = (active_trade.entry_price - price) / active_trade.entry_price * 100

            # Check trailing activation
            if profit_pct >= trailing_activation and not active_trade.trailing_active:
                active_trade.trailing_active = True

            exit_reason = None

            # Stop loss - always check
            if price >= active_trade.stop_loss:
                exit_reason = "stop_loss"

            # Trailing stop
            elif active_trade.trailing_active:
                trail_price = active_trade.best_price * (1 + trailing_pct / 100)
                if price >= trail_price:
                    exit_reason = "trailing_stop"

            # Lower BB exit - depends on option
            if exit_reason is None:
                if option == "A":
                    # No lower BB exit
                    pass
                elif option == "B":
                    # Use entry-time BB lower
                    if price <= active_trade.entry_bb_lower:
                        exit_reason = "lower_bb_locked"
                elif option == "C":
                    # Only if in profit
                    if profit_pct > 0 and price <= bb_lower:
                        exit_reason = "lower_bb_profit"

            if exit_reason:
                # Calculate P&L
                pnl = (active_trade.entry_price - price) * active_trade.size
                pnl_pct = (active_trade.entry_price - price) / active_trade.entry_price * 100

                # Subtract fees
                fees = notional * fee_rate * 2
                pnl -= fees

                trades.append(TradeResult(
                    entry_price=active_trade.entry_price,
                    exit_price=price,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    exit_reason=exit_reason,
                    hold_bars=hold_bars
                ))
                active_trade = None

        else:
            # Check entry
            if price > bb_upper and rsi > rsi_threshold:
                size = notional / price
                active_trade = Trade(
                    entry_idx=i,
                    entry_price=price,
                    entry_bb_lower=bb_lower,  # Lock BB at entry
                    size=size,
                    stop_loss=price * (1 + stop_loss_pct / 100),
                    best_price=price,
                )

    return trades

def analyze_results(trades: List[TradeResult], option: str, asset: str):
    """Analyze and print results."""
    if not trades:
        print(f"  {asset}: No trades")
        return {"trades": 0, "pnl": 0, "win_rate": 0}

    total_pnl = sum(t.pnl for t in trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    win_rate = wins / len(trades) * 100
    avg_hold = np.mean([t.hold_bars for t in trades])

    # Exit reason breakdown
    reasons = {}
    for t in trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1

    print(f"  {asset}: {len(trades)} trades, WR: {win_rate:.1f}%, PnL: ${total_pnl:.2f}, Avg Hold: {avg_hold:.0f} bars")
    for reason, count in reasons.items():
        print(f"       {reason}: {count}")

    return {"trades": len(trades), "pnl": total_pnl, "win_rate": win_rate}

def main():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    assets = ["BTC", "ETH", "SOL"]
    options = ["A", "B", "C"]

    print("=" * 70)
    print("RSI+BB Strategy Backtest - Exit Option Comparison")
    print("=" * 70)
    print("Entry: SHORT when price > Upper BB(20) AND RSI(7) > threshold")
    print("RSI thresholds:", RSI_THRESHOLDS)
    print("Stop Loss: 0.30%, Trailing: 0.20% @ 0.10% profit")
    print("Margin: $20 x 20x = $400 notional, Fee: 0.019% per side")
    print("=" * 70)
    print()
    print("Options:")
    print("  A: No lower BB exit (trailing + stop loss only)")
    print("  B: Lock BB at entry (exit at original lower BB)")
    print("  C: Lower BB exit only if trade is in profit")
    print()

    results = {opt: {"total_trades": 0, "total_pnl": 0, "total_wins": 0} for opt in options}

    for option in options:
        print(f"{'='*70}")
        print(f"OPTION {option}")
        print(f"{'='*70}")

        for asset in assets:
            file_path = os.path.join(data_dir, f"{asset}USDT_1m.csv")
            if not os.path.exists(file_path):
                print(f"  {asset}: Data file not found")
                continue

            df = pd.read_csv(file_path)
            trades = backtest_option(df, asset, option)
            stats = analyze_results(trades, option, asset)

            results[option]["total_trades"] += stats["trades"]
            results[option]["total_pnl"] += stats["pnl"]
            if stats["trades"] > 0:
                results[option]["total_wins"] += int(stats["win_rate"] * stats["trades"] / 100)

        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Option':<10} {'Trades':<10} {'Win Rate':<12} {'Total PnL':<15}")
    print("-" * 50)

    for option in options:
        r = results[option]
        wr = r["total_wins"] / r["total_trades"] * 100 if r["total_trades"] > 0 else 0
        print(f"{option:<10} {r['total_trades']:<10} {wr:.1f}%{'':<7} ${r['total_pnl']:.2f}")

    # Find best
    best = max(results.items(), key=lambda x: x[1]["total_pnl"])
    print()
    print(f"BEST OPTION: {best[0]} with ${best[1]['total_pnl']:.2f} total PnL")
    print("=" * 70)

if __name__ == "__main__":
    main()
