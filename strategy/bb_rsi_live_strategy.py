"""
RSI + Bollinger Band Live Trading Strategy

SHORT-only mean-reversion strategy:
- Entry: Price > Upper BB AND RSI > threshold
- Exit: Trailing stop (0.2% after 0.1% profit) OR Lower BB OR Stop Loss 0.3%
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Callable
from dataclasses import dataclass, field
from collections import deque
import pandas as pd
import numpy as np

from indicators import bollinger_bands, rsi

logger = logging.getLogger(__name__)


@dataclass
class BBRSITrade:
    """Active short trade."""
    entry_time: datetime
    entry_price: float
    size: float
    stop_loss_price: float
    best_price: float  # Lowest price seen (for trailing)
    trailing_active: bool = False
    entry_rsi: float = 0.0
    entry_bb_upper: float = 0.0


@dataclass
class TradeRecord:
    """Record of a completed trade."""
    entry_time: datetime
    exit_time: datetime
    side: str
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_pct: float
    exit_reason: str


@dataclass
class StrategyState:
    """Tracks the strategy's current state."""
    equity: float = 100.0
    daily_pnl: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    max_drawdown_pct: float = 0.0
    peak_equity: float = 100.0
    active_trade: Optional[BBRSITrade] = None
    trade_history: List[TradeRecord] = field(default_factory=list)


class BBRSILiveStrategy:
    """
    RSI + Bollinger Band live trading strategy.
    SHORT-only mean reversion with trailing stops.
    """

    def __init__(
        self,
        exchange_client,
        strategy_params: dict,
        risk_params: dict,
        candle_window: int = 30,
    ):
        self.exchange = exchange_client
        self.params = strategy_params
        self.risk = risk_params
        self.candle_window = candle_window

        # State
        initial_capital = risk_params.get("capital", {}).get("initial", 100)
        self.state = StrategyState(equity=initial_capital, peak_equity=initial_capital)
        self.candles: deque = deque(maxlen=candle_window)

        # Current indicator values
        self.current_bb_upper = 0.0
        self.current_bb_lower = 0.0
        self.current_bb_middle = 0.0
        self.current_rsi = 0.0

        # Callbacks
        self._on_trade_callbacks: List[Callable] = []

        logger.info("BBRSILiveStrategy initialized")
        logger.info(f"Strategy params: {self.params}")

    async def on_candle_close(self, candle: dict) -> Optional[int]:
        """Called when a new candle closes."""
        self.candles.append(candle)

        # Need enough data for indicators
        if len(self.candles) < self.candle_window:
            logger.debug(f"Building candle history: {len(self.candles)}/{self.candle_window}")
            return 0

        # Calculate indicators
        df = self._candles_to_df()
        self._calculate_indicators(df)

        current_price = candle["close"]

        # Check exits first if we have a position
        if self.state.active_trade:
            exit_reason = self._check_exit_conditions(current_price)
            if exit_reason:
                await self._close_position(exit_reason, current_price)
                return 0

        # Check for entry signal (SHORT only)
        if not self.state.active_trade:
            if self._check_entry_signal(current_price):
                await self._open_position(current_price)
                return -1

        return 0

    async def on_price_update(self, price: float) -> None:
        """Called on real-time price updates for stop monitoring."""
        if not self.state.active_trade:
            return

        trade = self.state.active_trade

        # Update best price (lowest for short)
        if price < trade.best_price:
            trade.best_price = price

        # Check trailing stop activation
        profit_pct = (trade.entry_price - price) / trade.entry_price * 100
        activation_pct = self.params.get("trailing_activation_pct", 0.10)

        if profit_pct >= activation_pct and not trade.trailing_active:
            trade.trailing_active = True
            logger.info(f"Trailing stop activated at {profit_pct:.2f}% profit")

        # Check stop loss
        if price >= trade.stop_loss_price:
            await self._close_position("stop_loss", price)
            return

        # Check trailing stop
        if trade.trailing_active:
            trailing_pct = self.params.get("trailing_stop_pct", 0.20) / 100
            trail_price = trade.best_price * (1 + trailing_pct)
            if price >= trail_price:
                await self._close_position("trailing_stop", price)
                return

    def _calculate_indicators(self, df: pd.DataFrame) -> None:
        """Calculate current indicator values."""
        bb_period = self.params.get("bb_period", 20)
        bb_std = self.params.get("bb_std", 2.0)
        rsi_period = self.params.get("rsi_period", 7)

        middle, upper, lower = bollinger_bands(df["close"], bb_period, bb_std)
        self.current_bb_middle = middle.iloc[-1]
        self.current_bb_upper = upper.iloc[-1]
        self.current_bb_lower = lower.iloc[-1]

        rsi_values = rsi(df["close"], rsi_period)
        self.current_rsi = rsi_values.iloc[-1]

    def _check_entry_signal(self, price: float) -> bool:
        """Check if SHORT entry conditions are met."""
        rsi_threshold = self.params.get("rsi_entry_threshold", 75)

        # SHORT: price > upper BB AND RSI > threshold
        if price > self.current_bb_upper and self.current_rsi > rsi_threshold:
            logger.info(f"Entry signal: price {price:.2f} > BB upper {self.current_bb_upper:.2f}, RSI {self.current_rsi:.1f} > {rsi_threshold}")
            return True
        return False

    def _check_exit_conditions(self, price: float) -> Optional[str]:
        """Check exit conditions for active trade."""
        trade = self.state.active_trade
        if not trade:
            return None

        # Lower BB exit
        if self.params.get("exit_on_lower_bb", True):
            if price <= self.current_bb_lower:
                return "lower_bb"

        return None

    async def _open_position(self, price: float) -> None:
        """Open a SHORT position."""
        # Calculate position size
        margin = self.params.get("margin_per_trade", 20.0)
        leverage = self.params.get("leverage", 20.0)
        notional = margin * leverage
        size = notional / price

        # Round size appropriately
        market = getattr(self.exchange, 'market', '')
        if 'SOL' in market:
            size = round(size, 2)
        else:
            size = round(size, 4)

        # Calculate stop loss
        stop_loss_pct = self.params.get("stop_loss_pct", 0.30) / 100
        stop_loss_price = price * (1 + stop_loss_pct)

        # Place order
        from exchange.base import OrderSide, OrderType
        try:
            order = await self.exchange.place_order(
                side=OrderSide.SELL,
                size=size,
                order_type=OrderType.MARKET,
            )

            entry_price = order.avg_fill_price or price

            self.state.active_trade = BBRSITrade(
                entry_time=datetime.now(),
                entry_price=entry_price,
                size=size,
                stop_loss_price=stop_loss_price,
                best_price=entry_price,
                entry_rsi=self.current_rsi,
                entry_bb_upper=self.current_bb_upper,
            )

            logger.info(f"OPENED SHORT: size={size:.4f}, entry={entry_price:.2f}, SL={stop_loss_price:.2f}, RSI={self.current_rsi:.1f}")

            for callback in self._on_trade_callbacks:
                await callback("open", self.state.active_trade)

        except Exception as e:
            logger.error(f"Failed to open position: {e}")

    async def _close_position(self, reason: str, price: float) -> None:
        """Close the active SHORT position."""
        if not self.state.active_trade:
            return

        trade = self.state.active_trade

        from exchange.base import OrderSide, OrderType
        try:
            order = await self.exchange.place_order(
                side=OrderSide.BUY,
                size=trade.size,
                order_type=OrderType.MARKET,
            )

            exit_price = order.avg_fill_price or price

            # Calculate P&L (SHORT: profit when price goes down)
            pnl = (trade.entry_price - exit_price) * trade.size
            pnl_pct = (trade.entry_price - exit_price) / trade.entry_price * 100

            # Subtract fees (Paradex 0.019% per side)
            fee_rate = 0.019 / 100
            notional = trade.size * trade.entry_price
            fees = notional * fee_rate * 2  # Entry + exit
            pnl -= fees

            # Record trade
            record = TradeRecord(
                entry_time=trade.entry_time,
                exit_time=datetime.now(),
                side="SHORT",
                entry_price=trade.entry_price,
                exit_price=exit_price,
                size=trade.size,
                pnl=pnl,
                pnl_pct=pnl_pct,
                exit_reason=reason,
            )
            self.state.trade_history.append(record)

            # Update state
            self.state.equity += pnl
            self.state.daily_pnl += pnl
            self.state.total_trades += 1

            if pnl >= 0:
                self.state.winning_trades += 1
            else:
                self.state.losing_trades += 1

            if self.state.equity > self.state.peak_equity:
                self.state.peak_equity = self.state.equity

            current_dd = (self.state.peak_equity - self.state.equity) / self.state.peak_equity * 100
            self.state.max_drawdown_pct = max(self.state.max_drawdown_pct, current_dd)

            logger.info(f"CLOSED SHORT: exit={exit_price:.2f}, P&L=${pnl:.2f} ({pnl_pct:.2f}%), reason={reason}")

            self.state.active_trade = None

            for callback in self._on_trade_callbacks:
                await callback("close", record)

        except Exception as e:
            logger.error(f"Failed to close position: {e}")

    def _candles_to_df(self) -> pd.DataFrame:
        """Convert candle deque to DataFrame."""
        return pd.DataFrame(list(self.candles))

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
            "max_drawdown_pct": self.state.max_drawdown_pct,
            "has_position": self.state.active_trade is not None,
            "is_paused": False,
            "current_rsi": self.current_rsi,
            "current_bb_upper": self.current_bb_upper,
            "current_bb_lower": self.current_bb_lower,
        }

    def add_trade_callback(self, callback: Callable) -> None:
        """Add callback for trade events."""
        self._on_trade_callbacks.append(callback)
