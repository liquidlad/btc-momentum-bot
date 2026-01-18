"""
RSI + BB Combo Strategy - Exit Optimization

Tests various exit triggers to reduce drawdown:
1. Different SL levels
2. Fixed take profits
3. Trailing stops
4. RSI-based exits
5. Time-based exits (max hold)
6. Partial exits (scale out)
7. Middle BB exit
8. Combined triggers
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import argparse
import itertools


def calculate_rsi(prices: np.ndarray, period: int = 7) -> np.ndarray:
    """Calculate RSI."""
    deltas = np.diff(prices, prepend=prices[0])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    gains_series = pd.Series(gains)
    losses_series = pd.Series(losses)

    avg_gains = gains_series.rolling(window=period, min_periods=period).mean().values
    avg_losses = losses_series.rolling(window=period, min_periods=period).mean().values

    rs = np.divide(avg_gains, avg_losses, out=np.ones_like(avg_gains), where=avg_losses != 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.nan_to_num(rsi, nan=50)

    return rsi


def calculate_bb(prices: np.ndarray, period: int = 20, std_mult: float = 2.0):
    """Calculate Bollinger Bands."""
    prices_series = pd.Series(prices)

    middle = prices_series.rolling(window=period, min_periods=period).mean().values
    std = prices_series.rolling(window=period, min_periods=period).std().values

    upper = middle + std_mult * std
    lower = middle - std_mult * std

    return lower, middle, upper


def run_backtest(df: pd.DataFrame, params: Dict) -> Dict:
    """
    Run backtest with various exit strategies.

    Exit modes:
    - 'sl_only': Pure stop loss, exit on lower BB
    - 'sl_tp': Stop loss + fixed take profit
    - 'trailing': Trailing stop
    - 'rsi_recenter': Exit when RSI drops below threshold
    - 'middle_bb': Exit when price crosses middle BB
    - 'time_exit': Exit after max bars
    - 'sl_tp_trailing': SL + TP + trailing after profit
    """
    prices = df['close'].values
    highs = df['high'].values if 'high' in df.columns else prices
    lows = df['low'].values if 'low' in df.columns else prices
    n = len(prices)

    # Parameters
    rsi_period = params.get('rsi_period', 7)
    rsi_entry = params.get('rsi_entry', 70)
    bb_period = params.get('bb_period', 20)
    bb_std = params.get('bb_std', 2.0)

    # Exit parameters
    stop_loss_pct = params.get('stop_loss_pct', 0.3)
    take_profit_pct = params.get('take_profit_pct', None)
    trailing_stop_pct = params.get('trailing_stop_pct', None)
    trailing_activation_pct = params.get('trailing_activation_pct', 0.1)  # Activate trailing after X% profit
    rsi_exit = params.get('rsi_exit', None)
    max_hold_bars = params.get('max_hold_bars', None)
    exit_on_middle_bb = params.get('exit_on_middle_bb', False)
    exit_on_lower_bb = params.get('exit_on_lower_bb', True)

    # Calculate indicators
    rsi = calculate_rsi(prices, rsi_period)
    lower_bb, middle_bb, upper_bb = calculate_bb(prices, bb_period, bb_std)

    # Simulate trades
    trades = []
    in_trade = False
    entry_price = 0
    entry_idx = 0
    best_price = 0  # For trailing stop (lowest price for short)
    trailing_active = False

    start_idx = max(bb_period, rsi_period) + 1

    for i in range(start_idx, n):
        if in_trade:
            should_exit = False
            exit_reason = None
            exit_price = prices[i]

            # Track best price for trailing (lowest for short = most profit)
            if prices[i] < best_price:
                best_price = prices[i]

            # Check if trailing should activate
            current_pnl_pct = (entry_price - prices[i]) / entry_price * 100
            if trailing_stop_pct and current_pnl_pct >= trailing_activation_pct:
                trailing_active = True

            # 1. Stop Loss (price went up)
            sl_price = entry_price * (1 + stop_loss_pct / 100)
            if highs[i] >= sl_price:
                should_exit = True
                exit_reason = "stop_loss"
                exit_price = sl_price

            # 2. Take Profit (price went down)
            if not should_exit and take_profit_pct:
                tp_price = entry_price * (1 - take_profit_pct / 100)
                if lows[i] <= tp_price:
                    should_exit = True
                    exit_reason = "take_profit"
                    exit_price = tp_price

            # 3. Trailing Stop
            if not should_exit and trailing_stop_pct and trailing_active:
                trail_price = best_price * (1 + trailing_stop_pct / 100)
                if prices[i] >= trail_price:
                    should_exit = True
                    exit_reason = "trailing_stop"
                    exit_price = trail_price

            # 4. RSI Recenter
            if not should_exit and rsi_exit and rsi[i] < rsi_exit:
                should_exit = True
                exit_reason = "rsi_exit"

            # 5. Middle BB exit
            if not should_exit and exit_on_middle_bb and prices[i] < middle_bb[i]:
                should_exit = True
                exit_reason = "middle_bb"

            # 6. Lower BB exit
            if not should_exit and exit_on_lower_bb and prices[i] < lower_bb[i]:
                should_exit = True
                exit_reason = "lower_bb"

            # 7. Time exit
            if not should_exit and max_hold_bars and (i - entry_idx) >= max_hold_bars:
                should_exit = True
                exit_reason = "time_exit"

            if should_exit:
                pnl_pct = (entry_price - exit_price) / entry_price * 100
                trades.append({
                    'pnl_pct': pnl_pct,
                    'exit_reason': exit_reason,
                    'hold_bars': i - entry_idx
                })
                in_trade = False
                trailing_active = False

        else:
            # Entry: price > upper BB AND RSI > threshold
            if not np.isnan(upper_bb[i]) and prices[i] > upper_bb[i] and rsi[i] > rsi_entry:
                in_trade = True
                entry_price = prices[i]
                entry_idx = i
                best_price = prices[i]
                trailing_active = False

    # Close open trade
    if in_trade:
        pnl_pct = (entry_price - prices[-1]) / entry_price * 100
        trades.append({
            'pnl_pct': pnl_pct,
            'exit_reason': 'end_of_data',
            'hold_bars': n - 1 - entry_idx
        })

    return analyze_trades(trades, df)


def analyze_trades(trades: List[Dict], df: pd.DataFrame) -> Dict:
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

    # Profit factor
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.001
    profit_factor = gross_profit / gross_loss

    # Exit reasons
    exit_reasons = {}
    for t in trades:
        r = t['exit_reason']
        exit_reasons[r] = exit_reasons.get(r, 0) + 1

    avg_hold = np.mean([t['hold_bars'] for t in trades])
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0

    # Risk-adjusted return
    risk_adj = total_pnl_usd / max(max_dd, 1)

    return {
        "total_trades": len(trades),
        "trades_per_day": len(trades) / trading_days,
        "win_rate": win_rate,
        "total_pnl_usd": total_pnl_usd,
        "max_drawdown_pct": max_dd,
        "avg_hold_bars": avg_hold,
        "profit_factor": profit_factor,
        "risk_adjusted": risk_adj,
        "avg_win_pct": avg_win,
        "avg_loss_pct": avg_loss,
        "exit_reasons": exit_reasons,
    }


def run_optimization(df: pd.DataFrame, asset: str, rsi_entry: int):
    """Test exit combinations."""

    # Base entry parameters (from previous optimization)
    base_params = {
        'bb_period': 20,
        'bb_std': 2.0,
        'rsi_period': 7,
        'rsi_entry': rsi_entry,
    }

    # Exit combinations to test
    exit_configs = [
        # Pure SL + lower BB (baseline)
        {'stop_loss_pct': 0.2, 'exit_on_lower_bb': True, 'name': 'SL0.2_LowerBB'},
        {'stop_loss_pct': 0.3, 'exit_on_lower_bb': True, 'name': 'SL0.3_LowerBB'},
        {'stop_loss_pct': 0.5, 'exit_on_lower_bb': True, 'name': 'SL0.5_LowerBB'},

        # SL + Fixed TP
        {'stop_loss_pct': 0.2, 'take_profit_pct': 0.3, 'exit_on_lower_bb': False, 'name': 'SL0.2_TP0.3'},
        {'stop_loss_pct': 0.2, 'take_profit_pct': 0.4, 'exit_on_lower_bb': False, 'name': 'SL0.2_TP0.4'},
        {'stop_loss_pct': 0.3, 'take_profit_pct': 0.4, 'exit_on_lower_bb': False, 'name': 'SL0.3_TP0.4'},
        {'stop_loss_pct': 0.3, 'take_profit_pct': 0.5, 'exit_on_lower_bb': False, 'name': 'SL0.3_TP0.5'},
        {'stop_loss_pct': 0.3, 'take_profit_pct': 0.6, 'exit_on_lower_bb': False, 'name': 'SL0.3_TP0.6'},
        {'stop_loss_pct': 0.4, 'take_profit_pct': 0.5, 'exit_on_lower_bb': False, 'name': 'SL0.4_TP0.5'},
        {'stop_loss_pct': 0.4, 'take_profit_pct': 0.6, 'exit_on_lower_bb': False, 'name': 'SL0.4_TP0.6'},

        # Trailing stop
        {'stop_loss_pct': 0.3, 'trailing_stop_pct': 0.2, 'trailing_activation_pct': 0.1, 'exit_on_lower_bb': True, 'name': 'SL0.3_Trail0.2@0.1'},
        {'stop_loss_pct': 0.3, 'trailing_stop_pct': 0.3, 'trailing_activation_pct': 0.2, 'exit_on_lower_bb': True, 'name': 'SL0.3_Trail0.3@0.2'},
        {'stop_loss_pct': 0.5, 'trailing_stop_pct': 0.2, 'trailing_activation_pct': 0.2, 'exit_on_lower_bb': True, 'name': 'SL0.5_Trail0.2@0.2'},
        {'stop_loss_pct': 0.5, 'trailing_stop_pct': 0.3, 'trailing_activation_pct': 0.3, 'exit_on_lower_bb': True, 'name': 'SL0.5_Trail0.3@0.3'},

        # RSI exit
        {'stop_loss_pct': 0.3, 'rsi_exit': 50, 'exit_on_lower_bb': False, 'name': 'SL0.3_RSI50'},
        {'stop_loss_pct': 0.3, 'rsi_exit': 40, 'exit_on_lower_bb': False, 'name': 'SL0.3_RSI40'},
        {'stop_loss_pct': 0.5, 'rsi_exit': 50, 'exit_on_lower_bb': False, 'name': 'SL0.5_RSI50'},
        {'stop_loss_pct': 0.5, 'rsi_exit': 40, 'exit_on_lower_bb': False, 'name': 'SL0.5_RSI40'},

        # RSI + Lower BB combo
        {'stop_loss_pct': 0.3, 'rsi_exit': 50, 'exit_on_lower_bb': True, 'name': 'SL0.3_RSI50_LowerBB'},
        {'stop_loss_pct': 0.3, 'rsi_exit': 40, 'exit_on_lower_bb': True, 'name': 'SL0.3_RSI40_LowerBB'},
        {'stop_loss_pct': 0.5, 'rsi_exit': 50, 'exit_on_lower_bb': True, 'name': 'SL0.5_RSI50_LowerBB'},

        # Middle BB exit
        {'stop_loss_pct': 0.3, 'exit_on_middle_bb': True, 'exit_on_lower_bb': False, 'name': 'SL0.3_MiddleBB'},
        {'stop_loss_pct': 0.5, 'exit_on_middle_bb': True, 'exit_on_lower_bb': False, 'name': 'SL0.5_MiddleBB'},

        # Time exit
        {'stop_loss_pct': 0.3, 'max_hold_bars': 30, 'exit_on_lower_bb': True, 'name': 'SL0.3_LowerBB_Max30'},
        {'stop_loss_pct': 0.3, 'max_hold_bars': 60, 'exit_on_lower_bb': True, 'name': 'SL0.3_LowerBB_Max60'},
        {'stop_loss_pct': 0.5, 'max_hold_bars': 30, 'exit_on_lower_bb': True, 'name': 'SL0.5_LowerBB_Max30'},

        # Tighter SL + TP combos (for lower DD)
        {'stop_loss_pct': 0.15, 'take_profit_pct': 0.2, 'exit_on_lower_bb': False, 'name': 'SL0.15_TP0.2'},
        {'stop_loss_pct': 0.15, 'take_profit_pct': 0.25, 'exit_on_lower_bb': False, 'name': 'SL0.15_TP0.25'},
        {'stop_loss_pct': 0.2, 'take_profit_pct': 0.25, 'exit_on_lower_bb': False, 'name': 'SL0.2_TP0.25'},
        {'stop_loss_pct': 0.2, 'take_profit_pct': 0.3, 'exit_on_lower_bb': False, 'name': 'SL0.2_TP0.3_pure'},

        # Combined: SL + TP + lower BB as backup
        {'stop_loss_pct': 0.3, 'take_profit_pct': 0.5, 'exit_on_lower_bb': True, 'name': 'SL0.3_TP0.5_LowerBB'},
        {'stop_loss_pct': 0.3, 'take_profit_pct': 0.6, 'exit_on_lower_bb': True, 'name': 'SL0.3_TP0.6_LowerBB'},
        {'stop_loss_pct': 0.4, 'take_profit_pct': 0.6, 'exit_on_lower_bb': True, 'name': 'SL0.4_TP0.6_LowerBB'},
    ]

    results = []

    for config in exit_configs:
        params = {**base_params, **config}
        name = config.pop('name', 'unnamed')

        metrics = run_backtest(df, params)

        if metrics.get("total_trades", 0) >= 10:
            metrics["name"] = name
            metrics["params"] = params
            results.append(metrics)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/BTCUSDT_1m.csv")
    parser.add_argument("--asset", default="BTC")
    parser.add_argument("--rsi-entry", type=int, default=70)
    args = parser.parse_args()

    print(f"Loading {args.data}...")
    df = pd.read_csv(args.data, parse_dates=["open_time"], index_col="open_time")
    trading_days = (df.index[-1] - df.index[0]).days or 1
    print(f"Data: {len(df)} candles over {trading_days} days")
    print(f"Asset: {args.asset}, RSI Entry: {args.rsi_entry}")
    print()

    results = run_optimization(df, args.asset, args.rsi_entry)

    if not results:
        print("No results!")
        return

    # Sort by risk-adjusted (PnL / MaxDD)
    results.sort(key=lambda x: x['risk_adjusted'], reverse=True)

    print("=" * 140)
    print(f"EXIT OPTIMIZATION RESULTS - {args.asset} (sorted by Risk-Adjusted = PnL/MaxDD)")
    print("=" * 140)
    print(f"{'#':<3} {'Strategy':<25} {'PnL$':<10} {'WR%':<8} {'MaxDD%':<8} {'PnL/DD':<8} {'Tr/Day':<8} {'Hold':<6} {'PF':<6} {'AvgWin':<8} {'AvgLoss':<8}")
    print("-" * 140)

    for i, r in enumerate(results[:25]):
        print(f"{i+1:<3} {r['name']:<25} ${r['total_pnl_usd']:<9.0f} {r['win_rate']:<8.1f} {r['max_drawdown_pct']:<8.1f} {r['risk_adjusted']:<8.1f} {r['trades_per_day']:<8.1f} {r['avg_hold_bars']:<6.0f} {r['profit_factor']:<6.2f} {r['avg_win_pct']:<8.2f} {r['avg_loss_pct']:<8.2f}")

    # Also show by pure PnL
    by_pnl = sorted(results, key=lambda x: x['total_pnl_usd'], reverse=True)

    print()
    print("=" * 140)
    print(f"TOP 10 BY PURE PNL - {args.asset}")
    print("=" * 140)
    print(f"{'#':<3} {'Strategy':<25} {'PnL$':<10} {'WR%':<8} {'MaxDD%':<8} {'PnL/DD':<8} {'Tr/Day':<8}")
    print("-" * 140)

    for i, r in enumerate(by_pnl[:10]):
        print(f"{i+1:<3} {r['name']:<25} ${r['total_pnl_usd']:<9.0f} {r['win_rate']:<8.1f} {r['max_drawdown_pct']:<8.1f} {r['risk_adjusted']:<8.1f} {r['trades_per_day']:<8.1f}")

    # Best low-drawdown options (MaxDD < 40%)
    low_dd = [r for r in results if r['max_drawdown_pct'] < 40 and r['total_pnl_usd'] > 0]
    if low_dd:
        low_dd.sort(key=lambda x: x['total_pnl_usd'], reverse=True)

        print()
        print("=" * 140)
        print(f"BEST LOW-DRAWDOWN OPTIONS (MaxDD < 40%) - {args.asset}")
        print("=" * 140)
        print(f"{'#':<3} {'Strategy':<25} {'PnL$':<10} {'WR%':<8} {'MaxDD%':<8} {'PnL/DD':<8} {'Tr/Day':<8}")
        print("-" * 140)

        for i, r in enumerate(low_dd[:10]):
            print(f"{i+1:<3} {r['name']:<25} ${r['total_pnl_usd']:<9.0f} {r['win_rate']:<8.1f} {r['max_drawdown_pct']:<8.1f} {r['risk_adjusted']:<8.1f} {r['trades_per_day']:<8.1f}")

    # Show exit reason breakdown for best risk-adjusted
    best = results[0]
    print()
    print("=" * 140)
    print(f"BEST RISK-ADJUSTED CONFIG: {best['name']}")
    print("=" * 140)
    print(f"PnL: ${best['total_pnl_usd']:.2f}")
    print(f"Win Rate: {best['win_rate']:.1f}%")
    print(f"Max Drawdown: {best['max_drawdown_pct']:.1f}%")
    print(f"Risk-Adjusted (PnL/DD): {best['risk_adjusted']:.1f}")
    print(f"Trades/Day: {best['trades_per_day']:.1f}")
    print(f"Avg Hold: {best['avg_hold_bars']:.0f} bars")
    print()
    print("Exit Reasons:")
    for k, v in best['exit_reasons'].items():
        pct = v / best['total_trades'] * 100
        print(f"  {k}: {v} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
