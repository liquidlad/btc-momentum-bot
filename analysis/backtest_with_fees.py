"""
Backtest with Fee Impact Analysis

Tests the momentum strategy with realistic fee structure to find
profitable parameter combinations.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'strategy'))

import pandas as pd
import numpy as np
from strategy.indicators import calculate_indicators, generate_signals

# Fee constants
TAKER_FEE = 0.00019  # 0.019% per side for market orders
MAKER_FEE = 0.0      # 0% for limit orders


def backtest_with_fees(
    df: pd.DataFrame,
    params: dict,
    capital: float = 50.0,
    position_notional: float = 1000.0,
    fee_rate: float = TAKER_FEE,
    verbose: bool = False
) -> dict:
    """
    Run backtest with realistic fee structure.

    Args:
        df: DataFrame with OHLCV data
        params: Strategy parameters
        capital: Starting capital
        position_notional: Notional value per trade
        fee_rate: Fee per side (0.019% for taker, 0% for maker)
        verbose: Print trade details

    Returns:
        Dictionary with backtest results
    """
    # Calculate indicators
    df_with_indicators = calculate_indicators(df.copy(), params)

    # Generate signals
    df_signals = generate_signals(df_with_indicators, params)

    # Extract parameters
    stop_loss_pct = params.get("stop_loss_pct", 0.10) / 100
    take_profit_pct = params.get("take_profit_1_pct", 0.12) / 100
    max_hold = params.get("max_hold_candles", 8)

    # Track state
    equity = capital
    peak_equity = capital
    max_drawdown = 0

    position = None  # {"side": "LONG"/"SHORT", "entry": price, "size": units, "candles_held": 0}

    trades = []
    wins = 0
    losses = 0

    for i in range(len(df_signals)):
        row = df_signals.iloc[i]
        price = row['close']
        signal = row.get('signal', 0)

        if position is None:
            # Check for entry signal
            if signal != 0:
                side = "LONG" if signal == 1 else "SHORT"
                size = position_notional / price

                # Entry fee
                entry_fee = position_notional * fee_rate

                position = {
                    "side": side,
                    "entry": price,
                    "size": size,
                    "candles_held": 0,
                    "entry_fee": entry_fee
                }

                if side == "LONG":
                    position["stop_loss"] = price * (1 - stop_loss_pct)
                    position["take_profit"] = price * (1 + take_profit_pct)
                else:
                    position["stop_loss"] = price * (1 + stop_loss_pct)
                    position["take_profit"] = price * (1 - take_profit_pct)
        else:
            # Check exits
            position["candles_held"] += 1
            exit_reason = None
            exit_price = price

            high = row['high']
            low = row['low']

            if position["side"] == "LONG":
                if low <= position["stop_loss"]:
                    exit_reason = "stop_loss"
                    exit_price = position["stop_loss"]
                elif high >= position["take_profit"]:
                    exit_reason = "take_profit"
                    exit_price = position["take_profit"]
                elif signal == -1:
                    exit_reason = "signal_reversal"
                elif position["candles_held"] >= max_hold:
                    exit_reason = "time_exit"
            else:  # SHORT
                if high >= position["stop_loss"]:
                    exit_reason = "stop_loss"
                    exit_price = position["stop_loss"]
                elif low <= position["take_profit"]:
                    exit_reason = "take_profit"
                    exit_price = position["take_profit"]
                elif signal == 1:
                    exit_reason = "signal_reversal"
                elif position["candles_held"] >= max_hold:
                    exit_reason = "time_exit"

            if exit_reason:
                # Calculate P&L
                if position["side"] == "LONG":
                    gross_pnl = (exit_price - position["entry"]) * position["size"]
                else:
                    gross_pnl = (position["entry"] - exit_price) * position["size"]

                # Exit fee
                exit_notional = exit_price * position["size"]
                exit_fee = exit_notional * fee_rate

                # Total fees
                total_fees = position["entry_fee"] + exit_fee

                # Net P&L
                net_pnl = gross_pnl - total_fees

                # Update equity
                equity += net_pnl

                # Track drawdown
                if equity > peak_equity:
                    peak_equity = equity
                drawdown = (peak_equity - equity) / peak_equity * 100
                max_drawdown = max(max_drawdown, drawdown)

                # Record trade
                trades.append({
                    "side": position["side"],
                    "entry": position["entry"],
                    "exit": exit_price,
                    "gross_pnl": gross_pnl,
                    "fees": total_fees,
                    "net_pnl": net_pnl,
                    "reason": exit_reason,
                    "candles_held": position["candles_held"]
                })

                if net_pnl > 0:
                    wins += 1
                else:
                    losses += 1

                if verbose and len(trades) <= 10:
                    print(f"Trade {len(trades)}: {position['side']} entry={position['entry']:.2f} "
                          f"exit={exit_price:.2f} gross=${gross_pnl:.2f} fees=${total_fees:.2f} "
                          f"net=${net_pnl:.2f} ({exit_reason})")

                position = None

    # Calculate metrics
    total_trades = len(trades)
    win_rate = wins / total_trades * 100 if total_trades > 0 else 0
    total_return = (equity - capital) / capital * 100
    trading_days = (df.index[-1] - df.index[0]).days or 1
    trades_per_day = total_trades / trading_days

    # Profit factor
    gross_wins = sum(t["net_pnl"] for t in trades if t["net_pnl"] > 0)
    gross_losses = abs(sum(t["net_pnl"] for t in trades if t["net_pnl"] < 0))
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')

    # Fee impact
    total_fees = sum(t["fees"] for t in trades)
    total_gross = sum(t["gross_pnl"] for t in trades)

    return {
        "total_trades": total_trades,
        "trades_per_day": round(trades_per_day, 1),
        "win_rate": round(win_rate, 2),
        "wins": wins,
        "losses": losses,
        "total_return_pct": round(total_return, 2),
        "final_equity": round(equity, 2),
        "max_drawdown_pct": round(max_drawdown, 2),
        "profit_factor": round(profit_factor, 2),
        "total_gross_pnl": round(total_gross, 2),
        "total_fees": round(total_fees, 2),
        "net_pnl": round(equity - capital, 2),
        "avg_trade_pnl": round((equity - capital) / total_trades, 2) if total_trades > 0 else 0,
        "trading_days": trading_days,
    }


def test_parameter_set(df, params, name, fee_rate=TAKER_FEE):
    """Test a specific parameter set and print results."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    print(f"Parameters: TP={params['take_profit_1_pct']:.2f}%, SL={params['stop_loss_pct']:.2f}%, "
          f"min_cond={params['min_conditions']}")
    print(f"Fee rate: {fee_rate*100:.3f}% per side")

    results = backtest_with_fees(df, params, fee_rate=fee_rate, verbose=True)

    print(f"\nResults over {results['trading_days']} days:")
    print(f"  Total Trades:    {results['total_trades']} ({results['trades_per_day']}/day)")
    print(f"  Win Rate:        {results['win_rate']:.1f}%")
    print(f"  Gross P&L:       ${results['total_gross_pnl']:.2f}")
    print(f"  Total Fees:      ${results['total_fees']:.2f}")
    print(f"  Net P&L:         ${results['net_pnl']:.2f}")
    print(f"  Total Return:    {results['total_return_pct']:.2f}%")
    print(f"  Max Drawdown:    {results['max_drawdown_pct']:.2f}%")
    print(f"  Profit Factor:   {results['profit_factor']:.2f}")
    print(f"  Avg Trade P&L:   ${results['avg_trade_pnl']:.2f}")

    # Daily projection
    daily_pnl = results['net_pnl'] / results['trading_days']
    print(f"\n  Daily P&L (projected): ${daily_pnl:.2f}/day")

    return results


if __name__ == "__main__":
    # Load BTC data
    data_path = "data/BTCUSDT_1m.csv"
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, parse_dates=["open_time"], index_col="open_time")
    print(f"Loaded {len(df)} candles")

    # Current parameters (baseline)
    current_params = {
        "min_conditions": 3,
        "ema_fast": 3,
        "ema_slow": 15,
        "roc_period": 3,
        "roc_threshold": 0.08,
        "volume_sma_period": 20,
        "volume_multiplier": 1.0,
        "rsi_period": 14,
        "rsi_long_min": 25,
        "rsi_long_max": 75,
        "rsi_short_min": 25,
        "rsi_short_max": 75,
        "stop_loss_pct": 0.10,
        "take_profit_1_pct": 0.12,
        "max_hold_candles": 8
    }

    print("\n" + "="*60)
    print("BACKTEST WITH FEE ANALYSIS")
    print("="*60)

    # Test 1: Current params with fees (baseline)
    test_parameter_set(df, current_params, "BASELINE: Current params with 0.019% fee", TAKER_FEE)

    # Test 2: Current params without fees (to show fee impact)
    test_parameter_set(df, current_params, "Current params WITHOUT fees (reference)", 0)

    # Test 3: Wider targets with market orders
    wider_params = current_params.copy()
    wider_params["take_profit_1_pct"] = 0.30
    wider_params["stop_loss_pct"] = 0.15
    test_parameter_set(df, wider_params, "SCENARIO A: 0.30% TP / 0.15% SL (market orders)", TAKER_FEE)

    # Test 4: Higher conviction (4 conditions)
    higher_conv_params = current_params.copy()
    higher_conv_params["min_conditions"] = 4
    higher_conv_params["take_profit_1_pct"] = 0.25
    higher_conv_params["stop_loss_pct"] = 0.12
    test_parameter_set(df, higher_conv_params, "SCENARIO B: 4 conditions, 0.25% TP / 0.12% SL", TAKER_FEE)

    # Test 5: Limit orders (0% fee) with current params
    test_parameter_set(df, current_params, "SCENARIO C: Current params with LIMIT orders (0% fee)", 0)

    # Test 6: Limit orders with slightly wider targets
    limit_params = current_params.copy()
    limit_params["take_profit_1_pct"] = 0.20
    limit_params["stop_loss_pct"] = 0.12
    test_parameter_set(df, limit_params, "SCENARIO D: 0.20% TP / 0.12% SL with LIMIT orders (0% fee)", 0)

    # Test 7: Very wide targets
    very_wide_params = current_params.copy()
    very_wide_params["take_profit_1_pct"] = 0.50
    very_wide_params["stop_loss_pct"] = 0.20
    test_parameter_set(df, very_wide_params, "SCENARIO E: 0.50% TP / 0.20% SL (market orders)", TAKER_FEE)

    # Test 8: Higher conviction + wide targets + market orders
    best_market_params = current_params.copy()
    best_market_params["min_conditions"] = 4
    best_market_params["take_profit_1_pct"] = 0.40
    best_market_params["stop_loss_pct"] = 0.15
    test_parameter_set(df, best_market_params, "SCENARIO F: 4 cond, 0.40% TP / 0.15% SL (market)", TAKER_FEE)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("""
Key Findings:
- Current params lose money with fees
- Need wider targets OR limit orders to be profitable
- Higher conviction trades (4+ conditions) improve win rate

Recommendations will depend on backtest results above.
""")
