"""
Live Trading Strategy Engine

Connects the backtested momentum strategy from Phase 1 with
real-time exchange execution from Phase 2.

This module:
1. Maintains a rolling window of candles from live data
2. Calculates indicators using the same logic as backtest
3. Generates signals using optimized parameters
4. Manages positions with stop-loss and take-profit
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Callable
from dataclasses import dataclass, field
from collections import deque
import pandas as pd
import numpy as np
import time

# Import strategy components from Phase 1
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from indicators import calculate_indicators, generate_signals

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Record of a completed trade."""
    entry_time: datetime
    exit_time: datetime
    side: str  # "LONG" or "SHORT"
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_pct: float
    exit_reason: str  # "stop_loss", "take_profit", "signal", "time_exit"


@dataclass
class ActiveTrade:
    """Currently active trade being managed."""
    entry_time: datetime
    side: str
    entry_price: float
    size: float
    stop_loss_price: float
    take_profit_price: float
    candles_held: int = 0
    highest_price: float = 0
    lowest_price: float = 0
    trailing_stop_active: bool = False
    partial_exit_done: bool = False


@dataclass
class StrategyState:
    """Tracks the strategy's current state."""
    equity: float = 200.0
    daily_pnl: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    consecutive_losses: int = 0
    max_drawdown_pct: float = 0.0
    peak_equity: float = 200.0
    active_trade: Optional[ActiveTrade] = None
    trade_history: List[TradeRecord] = field(default_factory=list)
    is_paused: bool = False
    pause_until: Optional[datetime] = None


class LiveStrategy:
    """
    Live trading strategy engine.

    Connects Phase 1 backtested strategy with Phase 2 exchange APIs.
    """

    def __init__(
        self,
        exchange_client,
        strategy_params: dict,
        risk_params: dict,
        candle_window: int = 20,
    ):
        """
        Initialize live strategy.

        Args:
            exchange_client: Exchange client (ParadexClient or LighterClient)
            strategy_params: Optimized strategy parameters from Phase 1
            risk_params: Risk management parameters
            candle_window: Number of candles to maintain for indicators
        """
        self.exchange = exchange_client
        self.params = strategy_params
        self.risk = risk_params
        self.candle_window = candle_window

        # State - use capital from risk config
        initial_capital = risk_params.get("capital", {}).get("initial", 50)
        self.state = StrategyState(equity=initial_capital, peak_equity=initial_capital)
        self.candles: deque = deque(maxlen=candle_window)
        self.last_signal = 0
        self.last_candle_time: Optional[datetime] = None

        # Callbacks for external monitoring
        self._on_signal_callbacks: List[Callable] = []
        self._on_trade_callbacks: List[Callable] = []
        self._on_state_change_callbacks: List[Callable] = []

        # Build indicator params from optimized values
        self.indicator_params = self._build_indicator_params()

        logger.info("LiveStrategy initialized")
        logger.info(f"Strategy params: {self.params}")

    def _build_indicator_params(self) -> dict:
        """Build indicator parameters from optimized values."""
        return {
            "ema_fast": self.params.get("ema_fast", 3),
            "ema_slow": self.params.get("ema_slow", 15),
            "roc_period": 3,
            "roc_threshold": self.params.get("roc_threshold", 0.08),
            "volume_sma_period": 20,
            "volume_multiplier": self.params.get("volume_multiplier", 1.0),
            "rsi_period": 14,
            "rsi_long_min": 25,
            "rsi_long_max": 75,
            "rsi_short_min": 25,
            "rsi_short_max": 75,
            "min_conditions": self.params.get("min_conditions", 3),
        }

    async def on_candle_close(self, candle: dict) -> Optional[int]:
        """
        Called when a new candle closes.

        Args:
            candle: Dict with open, high, low, close, volume, timestamp

        Returns:
            Signal: 1 (long), -1 (short), 0 (none)
        """
        # Add candle to history
        self.candles.append(candle)
        self.last_candle_time = datetime.fromtimestamp(candle["timestamp"] / 1000)

        # Update active trade candle count
        if self.state.active_trade:
            self.state.active_trade.candles_held += 1

        # Check if we have enough data for indicators
        if len(self.candles) < self.candle_window:
            logger.debug(f"Building candle history: {len(self.candles)}/{self.candle_window}")
            return 0

        # Check if paused
        if self.state.is_paused:
            if self.state.pause_until and datetime.now() < self.state.pause_until:
                logger.debug(f"Strategy paused until {self.state.pause_until}")
                return 0
            else:
                self.state.is_paused = False
                self.state.pause_until = None
                logger.info("Strategy resumed from pause")

        # Convert candles to DataFrame
        df = self._candles_to_df()

        # Calculate indicators
        df = calculate_indicators(df, self.indicator_params)

        # Generate signals
        df = generate_signals(df, self.indicator_params)

        # Get latest signal
        signal = int(df["signal"].iloc[-1])

        # Manage existing position or open new
        await self._process_signal(signal, candle)

        return signal

    async def on_price_update(self, price: float) -> None:
        """
        Called on real-time price updates (more frequent than candles).
        Used for stop-loss and take-profit monitoring.

        Args:
            price: Current market price
        """
        if not self.state.active_trade:
            return

        trade = self.state.active_trade

        # Update highest/lowest price tracking
        if trade.side == "LONG":
            trade.highest_price = max(trade.highest_price, price)
        else:
            trade.lowest_price = min(trade.lowest_price, price) if trade.lowest_price > 0 else price

        # Check stop loss
        if self._check_stop_loss(trade, price):
            await self._close_position("stop_loss", price)
            return

        # Check take profit
        if self._check_take_profit(trade, price):
            await self._close_position("take_profit", price)
            return

        # Update trailing stop if active
        if trade.trailing_stop_active:
            self._update_trailing_stop(trade, price)

    async def _process_signal(self, signal: int, candle: dict) -> None:
        """Process trading signal and manage positions."""
        current_price = candle["close"]

        # Check circuit breakers
        if not self._check_circuit_breakers():
            return

        # If we have an active trade
        if self.state.active_trade:
            trade = self.state.active_trade

            # Check time-based exit
            max_candles = self.risk.get("time_limits", {}).get("max_hold_candles", 10)
            if trade.candles_held >= max_candles:
                await self._close_position("time_exit", current_price)
                return

            # Check for signal reversal
            if (trade.side == "LONG" and signal == -1) or \
               (trade.side == "SHORT" and signal == 1):
                await self._close_position("signal_reversal", current_price)
                # Don't immediately open opposite position - wait for next candle

        # Open new position if no active trade and we have a signal
        elif signal != 0:
            await self._open_position(signal, current_price)

    async def _open_position(self, signal: int, price: float) -> None:
        """Open a new position."""
        side = "LONG" if signal == 1 else "SHORT"

        # Calculate position size (returns notional in USD)
        position_notional = self._calculate_position_size()
        if position_notional <= 0:
            logger.warning("Position size is 0, skipping trade")
            return

        # Convert notional to asset units (e.g., $1000 / $95000 = 0.0105 BTC)
        position_size = position_notional / price

        # Round to reasonable precision for the asset
        # BTC: 5 decimals, ETH: 4 decimals, SOL: 2 decimals
        position_size = round(position_size, 5)

        logger.info(f"Position sizing: ${position_notional:.2f} notional / ${price:.2f} = {position_size} units")

        # Calculate stop loss and take profit prices
        stop_loss_pct = self.params.get("stop_loss_pct", 0.10) / 100
        take_profit_pct = self.params.get("take_profit_1_pct", 0.12) / 100

        if side == "LONG":
            stop_loss_price = price * (1 - stop_loss_pct)
            take_profit_price = price * (1 + take_profit_pct)
        else:
            stop_loss_price = price * (1 + stop_loss_pct)
            take_profit_price = price * (1 - take_profit_pct)

        # Place order via exchange
        from exchange.base import OrderSide, OrderType
        order_side = OrderSide.BUY if side == "LONG" else OrderSide.SELL

        try:
            order = await self.exchange.place_order(
                side=order_side,
                size=position_size,
                order_type=OrderType.MARKET,
            )

            # Create active trade
            self.state.active_trade = ActiveTrade(
                entry_time=datetime.now(),
                side=side,
                entry_price=order.avg_fill_price or price,
                size=position_size,
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price,
                highest_price=price if side == "LONG" else 0,
                lowest_price=price if side == "SHORT" else float('inf'),
            )

            logger.info(f"OPENED {side} position: size={position_size:.4f}, entry={price:.2f}, SL={stop_loss_price:.2f}, TP={take_profit_price:.2f}")

            # Notify callbacks
            for callback in self._on_trade_callbacks:
                await callback("open", self.state.active_trade)

        except Exception as e:
            logger.error(f"Failed to open position: {e}")

    async def _close_position(self, reason: str, price: float) -> None:
        """Close the active position."""
        if not self.state.active_trade:
            return

        trade = self.state.active_trade

        # Place closing order
        from exchange.base import OrderSide, OrderType
        order_side = OrderSide.SELL if trade.side == "LONG" else OrderSide.BUY

        try:
            order = await self.exchange.place_order(
                side=order_side,
                size=trade.size,
                order_type=OrderType.MARKET,
            )

            exit_price = order.avg_fill_price or price

            # Calculate P&L
            if trade.side == "LONG":
                pnl = (exit_price - trade.entry_price) * trade.size
                pnl_pct = ((exit_price / trade.entry_price) - 1) * 100
            else:
                pnl = (trade.entry_price - exit_price) * trade.size
                pnl_pct = ((trade.entry_price / exit_price) - 1) * 100

            # Record trade
            trade_record = TradeRecord(
                entry_time=trade.entry_time,
                exit_time=datetime.now(),
                side=trade.side,
                entry_price=trade.entry_price,
                exit_price=exit_price,
                size=trade.size,
                pnl=pnl,
                pnl_pct=pnl_pct,
                exit_reason=reason,
            )
            self.state.trade_history.append(trade_record)

            # Update state
            self.state.equity += pnl
            self.state.daily_pnl += pnl
            self.state.total_trades += 1

            if pnl >= 0:
                self.state.winning_trades += 1
                self.state.consecutive_losses = 0
            else:
                self.state.losing_trades += 1
                self.state.consecutive_losses += 1

            # Update peak equity and drawdown
            if self.state.equity > self.state.peak_equity:
                self.state.peak_equity = self.state.equity

            current_dd = ((self.state.peak_equity - self.state.equity) / self.state.peak_equity) * 100
            self.state.max_drawdown_pct = max(self.state.max_drawdown_pct, current_dd)

            logger.info(f"CLOSED {trade.side} position: exit={exit_price:.2f}, P&L=${pnl:.2f} ({pnl_pct:.2f}%), reason={reason}")
            logger.info(f"Equity: ${self.state.equity:.2f}, Daily P&L: ${self.state.daily_pnl:.2f}")

            # Clear active trade
            self.state.active_trade = None

            # Notify callbacks
            for callback in self._on_trade_callbacks:
                await callback("close", trade_record)

        except Exception as e:
            logger.error(f"Failed to close position: {e}")

    def _calculate_position_size(self) -> float:
        """Calculate position size based on risk parameters."""
        risk_per_trade = self.state.equity * (self.risk.get("position_limits", {}).get("risk_per_trade_pct", 2.0) / 100)
        stop_loss_pct = self.params.get("stop_loss_pct", 0.10) / 100

        if stop_loss_pct <= 0:
            return 0

        size = risk_per_trade / stop_loss_pct

        # Apply leverage
        max_leverage = self.risk.get("leverage", {}).get("max", 50)
        max_position = self.state.equity * max_leverage

        # Cap at max position
        max_notional = self.risk.get("position_limits", {}).get("max_position_notional", 10000)
        size = min(size, max_position, max_notional)

        # Apply drawdown scaling
        if self.risk.get("drawdown_scaling", {}).get("enabled", True):
            current_dd = ((self.state.peak_equity - self.state.equity) / self.state.peak_equity) * 100
            thresholds = self.risk.get("drawdown_scaling", {}).get("thresholds", [])

            for threshold in sorted(thresholds, key=lambda x: x["drawdown_pct"], reverse=True):
                if current_dd >= threshold["drawdown_pct"]:
                    size *= threshold["position_multiplier"]
                    break

        return size

    def _check_circuit_breakers(self) -> bool:
        """Check if any circuit breakers are triggered."""
        breakers = self.risk.get("circuit_breakers", {})

        # Consecutive losses
        if self.state.consecutive_losses >= breakers.get("max_consecutive_losses", 5):
            logger.warning(f"Circuit breaker: {self.state.consecutive_losses} consecutive losses")
            self._pause_strategy(breakers.get("cooldown_after_losses_seconds", 300))
            return False

        # Daily drawdown
        daily_dd_pct = abs(self.state.daily_pnl / self.state.equity * 100) if self.state.daily_pnl < 0 else 0
        if daily_dd_pct >= breakers.get("max_daily_drawdown_pct", 25):
            logger.warning(f"Circuit breaker: Daily drawdown {daily_dd_pct:.1f}%")
            return False

        # Max trades
        if self.state.total_trades >= breakers.get("max_daily_trades", 500):
            logger.warning("Circuit breaker: Max daily trades reached")
            return False

        return True

    def _pause_strategy(self, seconds: int) -> None:
        """Pause the strategy for a specified duration."""
        self.state.is_paused = True
        self.state.pause_until = datetime.now() + timedelta(seconds=seconds)
        self.state.consecutive_losses = 0
        logger.info(f"Strategy paused for {seconds} seconds")

    def _check_stop_loss(self, trade: ActiveTrade, price: float) -> bool:
        """Check if stop loss should be triggered."""
        if trade.side == "LONG":
            return price <= trade.stop_loss_price
        else:
            return price >= trade.stop_loss_price

    def _check_take_profit(self, trade: ActiveTrade, price: float) -> bool:
        """Check if take profit should be triggered."""
        if trade.side == "LONG":
            return price >= trade.take_profit_price
        else:
            return price <= trade.take_profit_price

    def _update_trailing_stop(self, trade: ActiveTrade, price: float) -> None:
        """Update trailing stop loss."""
        trailing_pct = self.risk.get("take_profit", {}).get("trailing_distance_pct", 0.05) / 100

        if trade.side == "LONG":
            new_stop = trade.highest_price * (1 - trailing_pct)
            if new_stop > trade.stop_loss_price:
                trade.stop_loss_price = new_stop
        else:
            new_stop = trade.lowest_price * (1 + trailing_pct)
            if new_stop < trade.stop_loss_price:
                trade.stop_loss_price = new_stop

    def _candles_to_df(self) -> pd.DataFrame:
        """Convert candle deque to DataFrame."""
        data = list(self.candles)
        df = pd.DataFrame(data)

        # Ensure column names match what indicators expect
        column_mapping = {
            "timestamp": "open_time",
        }
        df = df.rename(columns=column_mapping)

        return df

    def get_stats(self) -> dict:
        """Get current strategy statistics."""
        win_rate = (self.state.winning_trades / self.state.total_trades * 100) if self.state.total_trades > 0 else 0

        return {
            "equity": self.state.equity,
            "daily_pnl": self.state.daily_pnl,
            "total_trades": self.state.total_trades,
            "winning_trades": self.state.winning_trades,
            "losing_trades": self.state.losing_trades,
            "win_rate": win_rate,
            "consecutive_losses": self.state.consecutive_losses,
            "max_drawdown_pct": self.state.max_drawdown_pct,
            "has_position": self.state.active_trade is not None,
            "is_paused": self.state.is_paused,
        }

    def add_trade_callback(self, callback: Callable) -> None:
        """Add callback for trade events."""
        self._on_trade_callbacks.append(callback)

    def reset_daily_stats(self) -> None:
        """Reset daily statistics (call at midnight)."""
        self.state.daily_pnl = 0
        logger.info("Daily stats reset")
