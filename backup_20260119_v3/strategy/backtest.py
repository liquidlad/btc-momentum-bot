"""
Backtesting engine for the momentum scalping strategy.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from indicators import calculate_indicators, generate_signals


@dataclass
class Trade:
    """Represents a single trade."""
    entry_time: pd.Timestamp
    entry_price: float
    direction: int  # 1 for long, -1 for short
    size: float  # notional size
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    exit_reason: Optional[str] = None
    candles_held: int = 0


class Backtester:
    """Backtesting engine for the momentum strategy."""

    def __init__(self, params: dict, capital: float = 200, leverage: float = 50):
        self.params = params
        self.capital = capital
        self.leverage = leverage
        self.trades: list[Trade] = []
        self.equity_curve: list[float] = []

    def calculate_position_size(self, current_equity: float) -> float:
        """Calculate position size based on risk parameters."""
        risk_amount = current_equity * (self.params["risk_per_trade_pct"] / 100)
        stop_pct = self.params["stop_loss_pct"] / 100
        position_size = risk_amount / stop_pct

        # Cap at max position
        max_position = min(
            self.params["max_position_notional"],
            current_equity * self.leverage
        )
        return min(position_size, max_position)

    def run(self, df: pd.DataFrame) -> dict:
        """
        Run backtest on historical data.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Dictionary with backtest results
        """
        # Calculate indicators and signals
        df = calculate_indicators(df, self.params)
        df = generate_signals(df, self.params)

        current_equity = self.capital
        current_trade: Optional[Trade] = None
        consecutive_losses = 0
        daily_loss = 0
        last_date = None

        self.equity_curve = [current_equity]

        for i in range(len(df)):
            row = df.iloc[i]
            current_date = row.name.date() if hasattr(row.name, 'date') else None

            # Reset daily loss counter on new day
            if current_date != last_date:
                daily_loss = 0
                last_date = current_date

            # Check circuit breakers
            if consecutive_losses >= 5:
                # Skip trading (in live would wait 30 min)
                consecutive_losses = 0  # Reset after "cooldown"
                continue

            if daily_loss >= current_equity * 0.10:
                # Stop trading for the day
                continue

            # If in a trade, check exit conditions
            if current_trade is not None:
                current_trade.candles_held += 1
                exit_price, exit_reason = self._check_exit(row, current_trade)

                if exit_price is not None:
                    # Close the trade
                    current_trade.exit_time = row.name
                    current_trade.exit_price = exit_price
                    current_trade.exit_reason = exit_reason

                    # Calculate PnL
                    if current_trade.direction == 1:  # Long
                        pnl_pct = (exit_price - current_trade.entry_price) / current_trade.entry_price
                    else:  # Short
                        pnl_pct = (current_trade.entry_price - exit_price) / current_trade.entry_price

                    current_trade.pnl = current_trade.size * pnl_pct
                    current_equity += current_trade.pnl

                    # Track consecutive losses
                    if current_trade.pnl < 0:
                        consecutive_losses += 1
                        daily_loss += abs(current_trade.pnl)
                    else:
                        consecutive_losses = 0

                    self.trades.append(current_trade)
                    current_trade = None

            # Check for new entry signal (only if not in a trade)
            if current_trade is None and row["signal"] != 0:
                direction = row["signal"]
                entry_price = row["close"]
                position_size = self.calculate_position_size(current_equity)

                # Calculate stop and take profit levels
                stop_pct = self.params["stop_loss_pct"] / 100
                tp1_pct = self.params["take_profit_1_pct"] / 100
                tp2_pct = self.params["take_profit_2_pct"] / 100

                if direction == 1:  # Long
                    stop_loss = entry_price * (1 - stop_pct)
                    take_profit_1 = entry_price * (1 + tp1_pct)
                    take_profit_2 = entry_price * (1 + tp2_pct)
                else:  # Short
                    stop_loss = entry_price * (1 + stop_pct)
                    take_profit_1 = entry_price * (1 - tp1_pct)
                    take_profit_2 = entry_price * (1 - tp2_pct)

                current_trade = Trade(
                    entry_time=row.name,
                    entry_price=entry_price,
                    direction=direction,
                    size=position_size,
                    stop_loss=stop_loss,
                    take_profit_1=take_profit_1,
                    take_profit_2=take_profit_2,
                )

            self.equity_curve.append(current_equity)

        # Close any remaining open trade at last price
        if current_trade is not None:
            last_row = df.iloc[-1]
            current_trade.exit_time = last_row.name
            current_trade.exit_price = last_row["close"]
            current_trade.exit_reason = "end_of_data"

            if current_trade.direction == 1:
                pnl_pct = (last_row["close"] - current_trade.entry_price) / current_trade.entry_price
            else:
                pnl_pct = (current_trade.entry_price - last_row["close"]) / current_trade.entry_price

            current_trade.pnl = current_trade.size * pnl_pct
            self.trades.append(current_trade)

        return self._calculate_metrics()

    def _check_exit(self, row: pd.Series, trade: Trade) -> tuple[Optional[float], Optional[str]]:
        """
        Check if trade should be exited.

        Returns:
            Tuple of (exit_price, exit_reason) or (None, None) if no exit
        """
        high = row["high"]
        low = row["low"]
        close = row["close"]

        if trade.direction == 1:  # Long
            # Check stop loss (using low of candle)
            if low <= trade.stop_loss:
                return trade.stop_loss, "stop_loss"

            # Check take profit (using high of candle)
            if high >= trade.take_profit_1:
                # Simplified: exit at TP1 (in reality would scale out)
                return trade.take_profit_1, "take_profit"

        else:  # Short
            # Check stop loss (using high of candle)
            if high >= trade.stop_loss:
                return trade.stop_loss, "stop_loss"

            # Check take profit (using low of candle)
            if low <= trade.take_profit_1:
                return trade.take_profit_1, "take_profit"

        # Time-based exit
        if trade.candles_held >= self.params["max_candles_in_trade"]:
            return close, "time_exit"

        # Exit if not in profit after N candles
        if trade.candles_held >= self.params.get("profit_check_candles", 10):
            if trade.direction == 1 and close <= trade.entry_price:
                return close, "no_profit_timeout"
            elif trade.direction == -1 and close >= trade.entry_price:
                return close, "no_profit_timeout"

        return None, None

    def _calculate_metrics(self) -> dict:
        """Calculate backtest performance metrics."""
        if not self.trades:
            return {"error": "No trades executed"}

        pnls = [t.pnl for t in self.trades if t.pnl is not None]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        total_trades = len(pnls)
        winning_trades = len(wins)
        losing_trades = len(losses)

        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0

        # Profit factor
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Total PnL
        total_pnl = sum(pnls)
        total_return = (total_pnl / self.capital) * 100

        # Max drawdown
        equity = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak * 100
        max_drawdown = np.max(drawdown)

        # Trade duration stats
        durations = [t.candles_held for t in self.trades]
        avg_duration = np.mean(durations) if durations else 0

        # Exit reason breakdown
        exit_reasons = {}
        for t in self.trades:
            reason = t.exit_reason or "unknown"
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

        # Calculate daily stats
        if self.trades:
            first_trade = self.trades[0].entry_time
            last_trade = self.trades[-1].exit_time or self.trades[-1].entry_time
            trading_days = (last_trade - first_trade).days or 1
            trades_per_day = total_trades / trading_days
        else:
            trades_per_day = 0

        # Sharpe ratio (simplified - assumes daily)
        if len(pnls) > 1:
            returns = np.array(pnls) / self.capital
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe = 0

        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": round(win_rate, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "avg_win_loss_ratio": round(abs(avg_win / avg_loss), 2) if avg_loss != 0 else float('inf'),
            "profit_factor": round(profit_factor, 2),
            "total_pnl": round(total_pnl, 2),
            "total_return_pct": round(total_return, 2),
            "max_drawdown_pct": round(max_drawdown, 2),
            "avg_trade_duration": round(avg_duration, 1),
            "trades_per_day": round(trades_per_day, 1),
            "sharpe_ratio": round(sharpe, 2),
            "final_equity": round(self.equity_curve[-1], 2),
            "exit_reasons": exit_reasons,
        }


# Default strategy parameters
DEFAULT_PARAMS = {
    # Moving Averages
    "ema_fast": 9,
    "ema_slow": 21,

    # Momentum
    "roc_period": 3,
    "roc_threshold": 0.15,

    # Volume
    "volume_sma_period": 20,
    "volume_multiplier": 1.5,

    # RSI
    "rsi_period": 14,
    "rsi_long_min": 40,
    "rsi_long_max": 70,
    "rsi_short_min": 30,
    "rsi_short_max": 60,

    # Risk Management
    "stop_loss_pct": 0.15,
    "take_profit_1_pct": 0.20,
    "take_profit_2_pct": 0.35,
    "trailing_stop_pct": 0.10,
    "trailing_activation_pct": 0.15,

    # Position
    "risk_per_trade_pct": 2.0,
    "max_position_notional": 2666,

    # Time
    "max_candles_in_trade": 20,
    "profit_check_candles": 10,
}


def run_backtest(data_path: str, params: dict = None, capital: float = 200, leverage: float = 50):
    """
    Run backtest on CSV data file.

    Args:
        data_path: Path to CSV file with OHLCV data
        params: Strategy parameters (uses defaults if None)
        capital: Starting capital
        leverage: Leverage multiplier

    Returns:
        Backtest results dictionary
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()

    # Load data
    df = pd.read_csv(data_path, parse_dates=["open_time"], index_col="open_time")

    # Run backtest
    bt = Backtester(params, capital, leverage)
    results = bt.run(df)

    return results, bt


if __name__ == "__main__":
    import sys

    # Check if data file exists
    data_file = "data/BTCUSDT_1m.csv"

    try:
        results, bt = run_backtest(data_file)

        print("\n" + "="*50)
        print("BACKTEST RESULTS")
        print("="*50)

        for key, value in results.items():
            if key != "exit_reasons":
                print(f"{key}: {value}")

        print("\nExit Reasons:")
        for reason, count in results.get("exit_reasons", {}).items():
            print(f"  {reason}: {count}")

    except FileNotFoundError:
        print(f"Data file not found: {data_file}")
        print("Run fetch_historical.py first to download data.")
        sys.exit(1)
