"""
Backtest: BB Short with Band-to-Band Exit

Entry: SHORT when price > upper Bollinger Band
Exit: When price < lower Bollinger Band (or SL hit)

Tests various SL values to find optimal.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
import argparse


@dataclass
class Trade:
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl_pct: float = 0.0


def calculate_bb(prices: np.ndarray, period: int = 20, std_mult: float = 2.0):
    """Calculate Bollinger Bands."""
    if len(prices) < period:
        return None, None, None

    sma = np.mean(prices[-period:])
    std = np.std(prices[-period:])

    upper = sma + std_mult * std
    lower = sma - std_mult * std

    return lower, sma, upper


def run_backtest(df: pd.DataFrame, bb_period: int = 20, bb_std: float = 2.0,
                 stop_loss_pct: float = None, leverage: float = 20.0,
                 margin_per_trade: float = 100.0):
    """
    Run backtest with band-to-band exit.

    Entry: SHORT when close > upper BB
    Exit: When close < lower BB (or SL if set)
    """
    trades: List[Trade] = []
    active_trade: Optional[Trade] = None

    prices = df['close'].values

    for i in range(bb_period, len(df)):
        current_price = prices[i]
        current_time = df.index[i]

        # Calculate BB
        lower_bb, sma, upper_bb = calculate_bb(prices[:i+1], bb_period, bb_std)

        if lower_bb is None:
            continue

        # Check exit first
        if active_trade:
            should_exit = False
            exit_reason = None

            # Check SL (price went up for short)
            if stop_loss_pct:
                sl_price = active_trade.entry_price * (1 + stop_loss_pct / 100)
                if current_price >= sl_price:
                    should_exit = True
                    exit_reason = "stop_loss"

            # Check band exit (price below lower BB)
            if not should_exit and current_price < lower_bb:
                should_exit = True
                exit_reason = "lower_band"

            if should_exit:
                active_trade.exit_time = current_time
                active_trade.exit_price = current_price
                active_trade.exit_reason = exit_reason
                # For short: profit when price goes down
                active_trade.pnl_pct = (active_trade.entry_price - current_price) / active_trade.entry_price * 100
                trades.append(active_trade)
                active_trade = None

        # Check entry (only if not in trade)
        if active_trade is None:
            if current_price > upper_bb:
                active_trade = Trade(
                    entry_time=current_time,
                    entry_price=current_price
                )

    # Close any open trade at end
    if active_trade:
        active_trade.exit_time = df.index[-1]
        active_trade.exit_price = prices[-1]
        active_trade.exit_reason = "end_of_data"
        active_trade.pnl_pct = (active_trade.entry_price - prices[-1]) / active_trade.entry_price * 100
        trades.append(active_trade)

    return trades


def analyze_trades(trades: List[Trade], trading_days: int, leverage: float = 20.0,
                   margin_per_trade: float = 100.0):
    """Analyze trade results."""
    if not trades:
        return {"error": "No trades"}

    notional = margin_per_trade * leverage

    wins = [t for t in trades if t.pnl_pct > 0]
    losses = [t for t in trades if t.pnl_pct <= 0]

    total_pnl_pct = sum(t.pnl_pct for t in trades)
    total_pnl_usd = sum(t.pnl_pct / 100 * notional for t in trades)

    win_rate = len(wins) / len(trades) * 100 if trades else 0

    avg_win = np.mean([t.pnl_pct for t in wins]) if wins else 0
    avg_loss = np.mean([t.pnl_pct for t in losses]) if losses else 0

    # Exit reason breakdown
    exit_reasons = {}
    for t in trades:
        reason = t.exit_reason or "unknown"
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

    # Calculate max drawdown
    equity = [margin_per_trade]
    for t in trades:
        pnl_usd = t.pnl_pct / 100 * notional
        equity.append(equity[-1] + pnl_usd)

    peak = equity[0]
    max_dd = 0
    for e in equity:
        if e > peak:
            peak = e
        dd = (peak - e) / peak * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    # Hold time analysis
    hold_times = [(t.exit_time - t.entry_time).total_seconds() / 60 for t in trades if t.exit_time]
    avg_hold_mins = np.mean(hold_times) if hold_times else 0

    return {
        "total_trades": len(trades),
        "trades_per_day": len(trades) / trading_days if trading_days > 0 else 0,
        "win_rate": win_rate,
        "total_pnl_pct": total_pnl_pct,
        "total_pnl_usd": total_pnl_usd,
        "avg_win_pct": avg_win,
        "avg_loss_pct": avg_loss,
        "max_drawdown_pct": max_dd,
        "final_equity": equity[-1],
        "avg_hold_mins": avg_hold_mins,
        "exit_reasons": exit_reasons,
    }


def main():
    parser = argparse.ArgumentParser(description="BB Band-to-Band Backtest")
    parser.add_argument("--data", default="data/BTCUSDT_1m.csv", help="Path to data file")
    parser.add_argument("--asset", default="BTC", help="Asset name for display")
    args = parser.parse_args()

    # Load data
    print(f"Loading {args.data}...")
    df = pd.read_csv(args.data, parse_dates=["open_time"], index_col="open_time")
    trading_days = (df.index[-1] - df.index[0]).days or 1

    print(f"Data: {len(df)} candles over {trading_days} days")
    print(f"Asset: {args.asset}")
    print()

    # Test various SL values (including no SL)
    sl_values = [None, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]

    print("=" * 100)
    print("BB BAND-TO-BAND STRATEGY BACKTEST")
    print("Entry: SHORT when price > upper BB(20, 2.0)")
    print("Exit: When price < lower BB (mean reversion complete)")
    print("=" * 100)
    print()
    print(f"{'SL%':<8} {'Trades':<8} {'Tr/Day':<8} {'WinRate':<10} {'TotalPnL%':<12} {'PnL$':<10} {'MaxDD%':<10} {'AvgHold':<10} {'SL_Exits':<10} {'Band_Exits':<10}")
    print("-" * 100)

    results = []

    for sl in sl_values:
        trades = run_backtest(df, bb_period=20, bb_std=2.0, stop_loss_pct=sl)
        metrics = analyze_trades(trades, trading_days)

        if "error" in metrics:
            continue

        sl_str = f"{sl:.1f}%" if sl else "None"
        sl_exits = metrics['exit_reasons'].get('stop_loss', 0)
        band_exits = metrics['exit_reasons'].get('lower_band', 0)

        print(f"{sl_str:<8} {metrics['total_trades']:<8} {metrics['trades_per_day']:<8.1f} {metrics['win_rate']:<10.1f} {metrics['total_pnl_pct']:<12.2f} ${metrics['total_pnl_usd']:<9.2f} {metrics['max_drawdown_pct']:<10.2f} {metrics['avg_hold_mins']:<10.1f} {sl_exits:<10} {band_exits:<10}")

        results.append({
            "sl_pct": sl,
            "metrics": metrics
        })

    print()
    print("=" * 100)

    # Find best by total PnL
    best = max(results, key=lambda x: x['metrics']['total_pnl_usd'])

    print()
    print("BEST CONFIGURATION (by total PnL):")
    print(f"  Stop Loss: {best['sl_pct']}%" if best['sl_pct'] else "  Stop Loss: None (band exit only)")
    print(f"  Total Trades: {best['metrics']['total_trades']}")
    print(f"  Win Rate: {best['metrics']['win_rate']:.1f}%")
    print(f"  Total PnL: ${best['metrics']['total_pnl_usd']:.2f}")
    print(f"  Max Drawdown: {best['metrics']['max_drawdown_pct']:.2f}%")
    print(f"  Avg Hold Time: {best['metrics']['avg_hold_mins']:.1f} minutes")
    print()

    # Also find best risk-adjusted (PnL / MaxDD)
    best_risk_adj = max(results, key=lambda x: x['metrics']['total_pnl_usd'] / max(x['metrics']['max_drawdown_pct'], 0.1))

    print("BEST RISK-ADJUSTED (PnL / MaxDD):")
    print(f"  Stop Loss: {best_risk_adj['sl_pct']}%" if best_risk_adj['sl_pct'] else "  Stop Loss: None")
    print(f"  Total PnL: ${best_risk_adj['metrics']['total_pnl_usd']:.2f}")
    print(f"  Max Drawdown: {best_risk_adj['metrics']['max_drawdown_pct']:.2f}%")


if __name__ == "__main__":
    main()
