"""
RSI + Bollinger Band Combo Strategy Backtest

Tests various combinations of RSI and BB for entries and exits.

Entry conditions (SHORT):
- Price > upper BB (overbought price)
- RSI > threshold (overbought momentum)

Exit conditions:
- RSI drops below exit threshold (momentum reversal)
- Price < lower BB (price reversal)
- Stop loss hit
- Combinations of above
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict
import argparse
import itertools


@dataclass
class Trade:
    entry_time: pd.Timestamp
    entry_price: float
    entry_rsi: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl_pct: float = 0.0


def calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate RSI."""
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gains = np.zeros(len(prices))
    avg_losses = np.zeros(len(prices))

    # First average
    avg_gains[period] = np.mean(gains[:period])
    avg_losses[period] = np.mean(losses[:period])

    # Smoothed averages
    for i in range(period + 1, len(prices)):
        avg_gains[i] = (avg_gains[i-1] * (period - 1) + gains[i-1]) / period
        avg_losses[i] = (avg_losses[i-1] * (period - 1) + losses[i-1]) / period

    rs = np.divide(avg_gains, avg_losses, out=np.zeros_like(avg_gains), where=avg_losses != 0)
    rsi = 100 - (100 / (1 + rs))
    rsi[:period] = 50  # Fill initial values

    return rsi


def calculate_bb(prices: np.ndarray, period: int = 20, std_mult: float = 2.0):
    """Calculate Bollinger Bands for entire series."""
    upper = np.full(len(prices), np.nan)
    lower = np.full(len(prices), np.nan)
    middle = np.full(len(prices), np.nan)

    for i in range(period, len(prices)):
        window = prices[i-period+1:i+1]
        sma = np.mean(window)
        std = np.std(window)
        middle[i] = sma
        upper[i] = sma + std_mult * std
        lower[i] = sma - std_mult * std

    return lower, middle, upper


def run_backtest(df: pd.DataFrame, params: Dict) -> Dict:
    """
    Run backtest with given parameters.

    params:
        bb_period: BB period
        bb_std: BB standard deviation multiplier
        rsi_period: RSI period
        rsi_entry: RSI threshold for entry (e.g., 70)
        rsi_exit: RSI threshold for exit (e.g., 50)
        exit_mode: 'rsi', 'lower_bb', 'rsi_or_bb', 'rsi_and_bb'
        stop_loss_pct: Stop loss percentage (None for no SL)
    """
    bb_period = params['bb_period']
    bb_std = params['bb_std']
    rsi_period = params['rsi_period']
    rsi_entry = params['rsi_entry']
    rsi_exit = params['rsi_exit']
    exit_mode = params['exit_mode']
    stop_loss_pct = params.get('stop_loss_pct')

    prices = df['close'].values

    # Calculate indicators
    rsi = calculate_rsi(prices, rsi_period)
    lower_bb, middle_bb, upper_bb = calculate_bb(prices, bb_period, bb_std)

    trades: List[Trade] = []
    active_trade: Optional[Trade] = None

    start_idx = max(bb_period, rsi_period) + 1

    for i in range(start_idx, len(df)):
        current_price = prices[i]
        current_rsi = rsi[i]
        current_time = df.index[i]
        current_upper = upper_bb[i]
        current_lower = lower_bb[i]

        # Check exit first
        if active_trade:
            should_exit = False
            exit_reason = None

            # Check SL
            if stop_loss_pct:
                sl_price = active_trade.entry_price * (1 + stop_loss_pct / 100)
                if current_price >= sl_price:
                    should_exit = True
                    exit_reason = "stop_loss"

            # Check exit conditions based on mode
            if not should_exit:
                if exit_mode == 'rsi':
                    # Exit when RSI drops below threshold
                    if current_rsi < rsi_exit:
                        should_exit = True
                        exit_reason = "rsi_exit"

                elif exit_mode == 'lower_bb':
                    # Exit when price < lower BB
                    if current_price < current_lower:
                        should_exit = True
                        exit_reason = "lower_bb"

                elif exit_mode == 'rsi_or_bb':
                    # Exit when RSI < threshold OR price < lower BB
                    if current_rsi < rsi_exit:
                        should_exit = True
                        exit_reason = "rsi_exit"
                    elif current_price < current_lower:
                        should_exit = True
                        exit_reason = "lower_bb"

                elif exit_mode == 'rsi_and_bb':
                    # Exit when RSI < threshold AND price < middle BB (conservative)
                    if current_rsi < rsi_exit and current_price < middle_bb[i]:
                        should_exit = True
                        exit_reason = "rsi_and_middle"

            if should_exit:
                active_trade.exit_time = current_time
                active_trade.exit_price = current_price
                active_trade.exit_reason = exit_reason
                active_trade.pnl_pct = (active_trade.entry_price - current_price) / active_trade.entry_price * 100
                trades.append(active_trade)
                active_trade = None

        # Check entry (only if not in trade)
        if active_trade is None:
            # Entry: Price > upper BB AND RSI > entry threshold
            if not np.isnan(current_upper) and current_price > current_upper and current_rsi > rsi_entry:
                active_trade = Trade(
                    entry_time=current_time,
                    entry_price=current_price,
                    entry_rsi=current_rsi
                )

    # Close any open trade at end
    if active_trade:
        active_trade.exit_time = df.index[-1]
        active_trade.exit_price = prices[-1]
        active_trade.exit_reason = "end_of_data"
        active_trade.pnl_pct = (active_trade.entry_price - prices[-1]) / active_trade.entry_price * 100
        trades.append(active_trade)

    return analyze_trades(trades, df)


def analyze_trades(trades: List[Trade], df: pd.DataFrame) -> Dict:
    """Analyze trade results."""
    trading_days = (df.index[-1] - df.index[0]).days or 1

    if not trades:
        return {"error": "No trades", "total_trades": 0}

    leverage = 20.0
    margin = 100.0
    notional = margin * leverage

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

    # Max drawdown
    equity = [margin]
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

    # Hold time
    hold_times = [(t.exit_time - t.entry_time).total_seconds() / 60 for t in trades if t.exit_time]
    avg_hold_mins = np.mean(hold_times) if hold_times else 0

    # Profit factor
    gross_profit = sum(t.pnl_pct for t in wins) if wins else 0
    gross_loss = abs(sum(t.pnl_pct for t in losses)) if losses else 0.001
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    return {
        "total_trades": len(trades),
        "trades_per_day": len(trades) / trading_days,
        "win_rate": win_rate,
        "total_pnl_pct": total_pnl_pct,
        "total_pnl_usd": total_pnl_usd,
        "avg_win_pct": avg_win,
        "avg_loss_pct": avg_loss,
        "max_drawdown_pct": max_dd,
        "avg_hold_mins": avg_hold_mins,
        "profit_factor": profit_factor,
        "exit_reasons": exit_reasons,
    }


def run_optimization(df: pd.DataFrame, asset: str):
    """Test all parameter combinations."""

    # Parameter ranges to test (reduced for speed)
    param_ranges = {
        'bb_period': [20],
        'bb_std': [2.0],
        'rsi_period': [7, 14],
        'rsi_entry': [65, 70, 75],
        'rsi_exit': [30, 40, 50],
        'exit_mode': ['rsi', 'lower_bb', 'rsi_or_bb'],
        'stop_loss_pct': [None, 0.3, 0.5, 1.0],
    }

    # Generate all combinations
    keys = list(param_ranges.keys())
    values = list(param_ranges.values())
    combinations = list(itertools.product(*values))

    print(f"Testing {len(combinations)} parameter combinations for {asset}...")
    print()

    results = []

    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))

        try:
            metrics = run_backtest(df, params)

            if "error" in metrics or metrics["total_trades"] < 10:
                continue

            # Only keep profitable strategies with reasonable metrics
            if metrics["total_pnl_usd"] > 0 and metrics["trades_per_day"] >= 1:
                metrics["params"] = params
                results.append(metrics)

        except Exception as e:
            continue

        if (i + 1) % 500 == 0:
            print(f"Progress: {i + 1}/{len(combinations)} - Found {len(results)} profitable configs", end="\r")

    print(f"\nFound {len(results)} profitable configurations")
    print()

    return results


def main():
    parser = argparse.ArgumentParser(description="RSI + BB Combo Strategy Backtest")
    parser.add_argument("--data", default="data/BTCUSDT_1m.csv", help="Path to data file")
    parser.add_argument("--asset", default="BTC", help="Asset name")
    args = parser.parse_args()

    # Load data
    print(f"Loading {args.data}...")
    df = pd.read_csv(args.data, parse_dates=["open_time"], index_col="open_time")
    trading_days = (df.index[-1] - df.index[0]).days or 1

    print(f"Data: {len(df)} candles over {trading_days} days")
    print(f"Asset: {args.asset}")
    print()
    print("=" * 120)
    print("RSI + BOLLINGER BAND COMBO STRATEGY")
    print("Entry: SHORT when price > upper BB AND RSI > threshold")
    print("Exit: Various modes (RSI recenter, lower BB, or combination)")
    print("=" * 120)
    print()

    # Run optimization
    results = run_optimization(df, args.asset)

    if not results:
        print("No profitable configurations found!")
        return

    # Sort by total PnL
    results.sort(key=lambda x: x["total_pnl_usd"], reverse=True)

    # Show top 20 by PnL
    print()
    print("=" * 120)
    print("TOP 20 BY TOTAL PNL")
    print("=" * 120)
    print(f"{'#':<3} {'PnL$':<10} {'WinRate':<8} {'Tr/Day':<8} {'MaxDD%':<8} {'PF':<6} {'AvgHold':<8} {'BB':<10} {'RSI':<12} {'Entry':<6} {'Exit':<6} {'ExitMode':<12} {'SL':<6}")
    print("-" * 120)

    for i, r in enumerate(results[:20]):
        p = r['params']
        bb_str = f"{p['bb_period']}/{p['bb_std']}"
        rsi_str = f"{p['rsi_period']}p"
        sl_str = f"{p['stop_loss_pct']}%" if p['stop_loss_pct'] else "None"

        print(f"{i+1:<3} ${r['total_pnl_usd']:<9.0f} {r['win_rate']:<8.1f} {r['trades_per_day']:<8.1f} {r['max_drawdown_pct']:<8.1f} {r['profit_factor']:<6.2f} {r['avg_hold_mins']:<8.0f} {bb_str:<10} {rsi_str:<12} {p['rsi_entry']:<6} {p['rsi_exit']:<6} {p['exit_mode']:<12} {sl_str:<6}")

    # Sort by win rate (min 5 trades/day)
    high_wr = [r for r in results if r['trades_per_day'] >= 5]
    if high_wr:
        high_wr.sort(key=lambda x: x["win_rate"], reverse=True)

        print()
        print("=" * 120)
        print("TOP 10 BY WIN RATE (min 5 trades/day)")
        print("=" * 120)
        print(f"{'#':<3} {'WinRate':<8} {'PnL$':<10} {'Tr/Day':<8} {'MaxDD%':<8} {'PF':<6} {'BB':<10} {'RSI':<8} {'Entry':<6} {'Exit':<6} {'ExitMode':<12} {'SL':<6}")
        print("-" * 120)

        for i, r in enumerate(high_wr[:10]):
            p = r['params']
            bb_str = f"{p['bb_period']}/{p['bb_std']}"
            rsi_str = f"{p['rsi_period']}p"
            sl_str = f"{p['stop_loss_pct']}%" if p['stop_loss_pct'] else "None"

            print(f"{i+1:<3} {r['win_rate']:<8.1f} ${r['total_pnl_usd']:<9.0f} {r['trades_per_day']:<8.1f} {r['max_drawdown_pct']:<8.1f} {r['profit_factor']:<6.2f} {bb_str:<10} {rsi_str:<8} {p['rsi_entry']:<6} {p['rsi_exit']:<6} {p['exit_mode']:<12} {sl_str:<6}")

    # Sort by risk-adjusted (PnL / MaxDD)
    risk_adj = sorted(results, key=lambda x: x["total_pnl_usd"] / max(x["max_drawdown_pct"], 1), reverse=True)

    print()
    print("=" * 120)
    print("TOP 10 RISK-ADJUSTED (PnL / MaxDD)")
    print("=" * 120)
    print(f"{'#':<3} {'PnL/DD':<8} {'PnL$':<10} {'WinRate':<8} {'MaxDD%':<8} {'Tr/Day':<8} {'BB':<10} {'RSI':<8} {'Entry':<6} {'Exit':<6} {'ExitMode':<12} {'SL':<6}")
    print("-" * 120)

    for i, r in enumerate(risk_adj[:10]):
        p = r['params']
        bb_str = f"{p['bb_period']}/{p['bb_std']}"
        rsi_str = f"{p['rsi_period']}p"
        sl_str = f"{p['stop_loss_pct']}%" if p['stop_loss_pct'] else "None"
        ratio = r["total_pnl_usd"] / max(r["max_drawdown_pct"], 1)

        print(f"{i+1:<3} {ratio:<8.1f} ${r['total_pnl_usd']:<9.0f} {r['win_rate']:<8.1f} {r['max_drawdown_pct']:<8.1f} {r['trades_per_day']:<8.1f} {bb_str:<10} {rsi_str:<8} {p['rsi_entry']:<6} {p['rsi_exit']:<6} {p['exit_mode']:<12} {sl_str:<6}")

    # Best overall
    best = results[0]
    print()
    print("=" * 120)
    print("BEST CONFIGURATION (by PnL)")
    print("=" * 120)
    print(f"Total PnL: ${best['total_pnl_usd']:.2f}")
    print(f"Win Rate: {best['win_rate']:.1f}%")
    print(f"Trades/Day: {best['trades_per_day']:.1f}")
    print(f"Max Drawdown: {best['max_drawdown_pct']:.1f}%")
    print(f"Profit Factor: {best['profit_factor']:.2f}")
    print(f"Avg Hold: {best['avg_hold_mins']:.0f} mins")
    print()
    print("Parameters:")
    for k, v in best['params'].items():
        print(f"  {k}: {v}")
    print()
    print("Exit Reasons:")
    for reason, count in best['exit_reasons'].items():
        print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
