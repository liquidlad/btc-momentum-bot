"""
RSI + Bollinger Band Strategy with Trailing Stop

Entry: SHORT when price > upper BB AND RSI > threshold
Exit: Trailing stop (0.2% after 0.1% profit) OR lower BB OR stop loss

Optimized parameters from backtest:
- BTC: RSI > 75, SL 0.3%, Trail 0.2%@0.1%
- ETH: RSI > 60, SL 0.3%, Trail 0.2%@0.1%
- SOL: RSI > 65, SL 0.3%, Trail 0.2%@0.1%

90-day backtest results:
- BTC: $1,013 PnL, 56% WR, 41% MaxDD
- ETH: $1,371 PnL, 52% WR, 44% MaxDD
- SOL: $1,792 PnL, 52% WR, 51% MaxDD
"""

import asyncio
import logging
import csv
import os
import time
from datetime import datetime
from typing import Optional, Dict
from dataclasses import dataclass
from collections import deque
import numpy as np
import sys

from exchange.lighter_client import NetworkError

# Add shared_prices to path for price_reader
sys.path.insert(0, r"C:\Users\eliha\shared_prices")
from price_reader import get_price, get_bbo, is_price_fresh

logger = logging.getLogger(__name__)

# Shared price config
PRICE_MAX_AGE = 15  # seconds - skip trading if price older than this

# Network error handling
MAX_CONSECUTIVE_NETWORK_ERRORS = 10  # Max before giving up
NETWORK_ERROR_BACKOFF_BASE = 2.0  # Base seconds for exponential backoff
NETWORK_ERROR_BACKOFF_MAX = 60.0  # Max backoff seconds

# Trade history CSV path
TRADE_HISTORY_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(TRADE_HISTORY_DIR, exist_ok=True)


@dataclass
class RSIBBTrade:
    """Active short trade."""
    asset: str
    entry_time: datetime
    entry_price: float
    size: float
    stop_loss_price: float
    best_price: float  # Lowest price seen (for trailing)
    trailing_active: bool = False


@dataclass
class RSIBBConfig:
    """Configuration for RSI+BB strategy."""
    # Bollinger Band settings
    bb_period: int = 20
    bb_std: float = 2.0

    # RSI settings
    rsi_period: int = 7

    # Exit settings
    stop_loss_pct: float = 0.30
    trailing_stop_pct: float = 0.20
    trailing_activation_pct: float = 0.10  # Activate trailing after 0.1% profit

    # Position sizing
    margin_per_trade: float = 100.0
    leverage: float = 20.0


class RSIBBStrategy:
    """
    RSI + Bollinger Band Strategy with Trailing Stop.

    Entry: SHORT when price > upper BB AND RSI > threshold
    Exit: Trailing stop OR lower BB OR stop loss
    """

    def __init__(
        self,
        exchange_clients: Dict[str, any],
        config: RSIBBConfig = None,
        rsi_entry_overrides: Dict[str, int] = None,
    ):
        self.clients = exchange_clients
        self.config = config or RSIBBConfig()

        # Per-asset RSI entry thresholds (optimized)
        self.rsi_entry = rsi_entry_overrides or {
            "BTC": 75,
            "ETH": 60,
            "SOL": 65,
        }

        # Price history for indicators
        self.price_history: Dict[str, deque] = {
            asset: deque(maxlen=max(self.config.bb_period, self.config.rsi_period) + 10)
            for asset in exchange_clients.keys()
        }

        # Active trades
        self.active_trades: Dict[str, Optional[RSIBBTrade]] = {
            asset: None for asset in exchange_clients.keys()
        }

        # Track last price collection time per asset (for consistent BB calculation)
        self._last_price_time: Dict[str, float] = {
            asset: 0 for asset in exchange_clients.keys()
        }
        self._price_collection_interval = 10  # Collect price every 10 sec for BB

        # Stats
        self.stats = {
            asset: {
                "trades": 0, "wins": 0, "losses": 0, "pnl": 0.0,
                "trailing_exits": 0, "lower_bb_exits": 0, "sl_exits": 0
            }
            for asset in exchange_clients.keys()
        }

        # Halted assets - when True, no orders will be placed for that asset
        self.halted: Dict[str, bool] = {
            asset: False for asset in exchange_clients.keys()
        }

        self.running = False

        # Trade history CSV
        self.trade_history_file = os.path.join(TRADE_HISTORY_DIR, "rsi_bb_trades.csv")
        self._init_trade_history()

    def _init_trade_history(self):
        """Initialize trade history CSV."""
        if not os.path.exists(self.trade_history_file):
            with open(self.trade_history_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'asset', 'entry_time', 'exit_time', 'entry_price',
                    'exit_price', 'best_price', 'size', 'pnl_pct', 'pnl_usd',
                    'exit_reason', 'sl_price', 'hold_seconds', 'entry_rsi'
                ])

    def _log_trade(self, asset: str, trade: RSIBBTrade, exit_price: float,
                   pnl_pct: float, pnl_usd: float, exit_reason: str, entry_rsi: float):
        """Log completed trade to CSV."""
        try:
            hold_seconds = (datetime.now() - trade.entry_time).total_seconds()
            with open(self.trade_history_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    asset,
                    trade.entry_time.isoformat(),
                    datetime.now().isoformat(),
                    trade.entry_price,
                    exit_price,
                    trade.best_price,
                    trade.size,
                    round(pnl_pct, 4),
                    round(pnl_usd, 2),
                    exit_reason,
                    trade.stop_loss_price,
                    round(hold_seconds, 1),
                    round(entry_rsi, 1)
                ])
        except Exception as e:
            logger.error(f"Failed to log trade: {e}")

    def calculate_rsi(self, prices: list) -> float:
        """Calculate RSI."""
        if len(prices) < self.config.rsi_period + 1:
            return 50.0  # Neutral

        prices_arr = np.array(prices)
        deltas = np.diff(prices_arr)

        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        recent_gains = gains[-self.config.rsi_period:]
        recent_losses = losses[-self.config.rsi_period:]

        avg_gain = np.mean(recent_gains)
        avg_loss = np.mean(recent_losses)

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def calculate_bb(self, prices: list) -> tuple:
        """Calculate Bollinger Bands."""
        if len(prices) < self.config.bb_period:
            return None, None, None

        recent = list(prices)[-self.config.bb_period:]
        sma = np.mean(recent)
        std = np.std(recent)

        upper = sma + self.config.bb_std * std
        lower = sma - self.config.bb_std * std

        return lower, sma, upper

    async def check_entry(self, asset: str, current_price: float) -> Optional[float]:
        """Check if we should enter. Returns RSI if entry, None otherwise."""
        if self.active_trades[asset] is not None:
            return None

        prices = list(self.price_history[asset])
        if len(prices) < max(self.config.bb_period, self.config.rsi_period + 1):
            return None

        _, _, upper_bb = self.calculate_bb(prices)
        rsi = self.calculate_rsi(prices)

        rsi_threshold = self.rsi_entry.get(asset, 70)

        if upper_bb and current_price > upper_bb and rsi > rsi_threshold:
            logger.info(f"{asset}: Price {current_price:.2f} > Upper BB {upper_bb:.2f}, RSI {rsi:.1f} > {rsi_threshold} - SHORT SIGNAL")
            return rsi

        return None

    async def check_exit(self, asset: str, current_price: float) -> Optional[str]:
        """Check exit conditions. Returns exit reason or None."""
        trade = self.active_trades[asset]
        if trade is None:
            return None

        # Update best price (lowest for short = most profit)
        if current_price < trade.best_price:
            trade.best_price = current_price

        # Calculate current PnL
        pnl_pct = (trade.entry_price - current_price) / trade.entry_price * 100

        # Check if trailing should activate
        if not trade.trailing_active and pnl_pct >= self.config.trailing_activation_pct:
            trade.trailing_active = True
            logger.info(f"{asset}: Trailing stop activated at {pnl_pct:.2f}% profit")

        # 1. Stop Loss
        if current_price >= trade.stop_loss_price:
            return "stop_loss"

        # 2. Trailing Stop (if active)
        if trade.trailing_active:
            trail_price = trade.best_price * (1 + self.config.trailing_stop_pct / 100)
            if current_price >= trail_price:
                return "trailing_stop"

        # 3. Lower BB exit
        prices = list(self.price_history[asset])
        if len(prices) >= self.config.bb_period:
            lower_bb, _, _ = self.calculate_bb(prices)
            if lower_bb and current_price < lower_bb:
                return "lower_bb"

        return None

    async def _check_position_sanity(self, asset: str, current_price: float) -> bool:
        """
        Check if position size is within safe limits.
        If position exceeds 2x expected size:
        1. Close ALL positions on ALL assets (using exchange sizes)
        2. Halt ALL trading
        Returns False if halted.
        """
        if self.halted.get(asset, False):
            return False  # Already halted

        try:
            client = self.clients[asset]
            positions = await client.get_positions()

            # Expected max size based on config
            expected_size = (self.config.margin_per_trade * self.config.leverage) / current_price
            max_allowed = expected_size * 2  # 2x threshold

            for pos in positions:
                if pos.size > max_allowed:
                    logger.error(f"{asset}: === EMERGENCY HALT ===")
                    logger.error(f"{asset}: Position size {pos.size:.6f} {pos.side} exceeds max allowed {max_allowed:.6f}")
                    logger.error(f"{asset}: Expected size: {expected_size:.6f}")
                    logger.error(f"{asset}: Current price: {current_price:.2f}")
                    logger.error(f"{asset}: Unrealized PnL: ${pos.unrealized_pnl:.2f}")

                    # Close ALL positions on ALL assets
                    logger.error(f"=== CLOSING ALL POSITIONS ===")
                    await self._emergency_close_all()

                    # Halt ALL assets
                    logger.error(f"=== HALTING ALL TRADING ===")
                    for a in self.halted.keys():
                        self.halted[a] = True

                    # Clear all active trades
                    for a in self.active_trades.keys():
                        self.active_trades[a] = None

                    logger.error(f"All positions closed, all assets halted. Restart bot to resume.")
                    return False

            return True

        except Exception as e:
            logger.error(f"{asset}: Error checking position sanity: {e}")
            return True  # Don't halt on error, but log it

    async def _emergency_close_all(self):
        """Emergency close ALL positions on ALL assets using exchange sizes."""
        from exchange.base import OrderSide, OrderType

        for asset, client in self.clients.items():
            try:
                positions = await client.get_positions()
                for pos in positions:
                    if pos.size > 0.0001:
                        logger.error(f"{asset}: Closing {pos.size:.6f} {pos.side}...")
                        close_side = OrderSide.SELL if pos.side == "LONG" else OrderSide.BUY
                        try:
                            await client.place_order(
                                side=close_side,
                                size=pos.size,  # Use EXCHANGE size
                                order_type=OrderType.MARKET,
                                slippage_pct=1.0,  # 1% slippage for emergency
                            )
                            logger.error(f"{asset}: Close order sent for {pos.size:.6f} {pos.side}")
                        except Exception as e:
                            logger.error(f"{asset}: Failed to close {pos.side}: {e}")
            except Exception as e:
                logger.error(f"{asset}: Failed to get positions: {e}")

    async def enter_short(self, asset: str, current_price: float, entry_rsi: float):
        """Enter a short position. Only called when update() confirms no position exists."""
        # Check if halted
        if self.halted.get(asset, False):
            logger.warning(f"{asset}: Trading halted - skipping entry")
            return

        client = self.clients[asset]

        notional = self.config.margin_per_trade * self.config.leverage
        size = notional / current_price

        sl_price = current_price * (1 + self.config.stop_loss_pct / 100)

        # CRITICAL: Set active_trades BEFORE placing order to prevent duplicate entries
        # If order fails completely, we clear it. If order status is uncertain, we keep it
        # to prevent placing duplicate orders that could drain margin.
        self.active_trades[asset] = RSIBBTrade(
            asset=asset,
            entry_time=datetime.now(),
            entry_price=current_price,  # Will update after fill
            size=size,
            stop_loss_price=sl_price,
            best_price=current_price,
            trailing_active=False,
        )
        self.active_trades[asset]._entry_rsi = entry_rsi

        try:
            from exchange.base import OrderSide, OrderType

            order = await client.place_order(
                side=OrderSide.SELL,
                size=size,
                order_type=OrderType.MARKET,
            )

            fill_price = order.avg_fill_price or current_price

            # Update with actual fill price
            self.active_trades[asset].entry_price = fill_price
            self.active_trades[asset].best_price = fill_price

            # CRITICAL: Verify actual position size from exchange
            # The filled size may differ from requested size
            await asyncio.sleep(0.5)  # Brief delay for position to update
            try:
                positions = await client.get_positions()
                actual_size = None
                for pos in positions:
                    if pos.side == "SHORT" and pos.size > 0:
                        actual_size = pos.size
                        break

                if actual_size is not None and abs(actual_size - size) > 0.001:
                    logger.warning(
                        f"{asset}: POSITION SIZE MISMATCH - Expected: {size:.4f}, Actual: {actual_size:.4f}"
                    )
                    # Update tracked size to actual size
                    self.active_trades[asset].size = actual_size
                    logger.info(f"{asset}: Updated tracked size to actual: {actual_size:.4f}")
                elif actual_size is not None:
                    logger.info(f"{asset}: Position verified - size: {actual_size:.4f}")
                else:
                    logger.warning(f"{asset}: Could not verify position - no SHORT found on exchange")
            except Exception as ve:
                logger.warning(f"{asset}: Could not verify position: {ve}")

            logger.info(
                f"{asset}: ENTERED SHORT @ {fill_price:.2f}, RSI: {entry_rsi:.1f}, "
                f"Size: ${notional:.0f} ({self.active_trades[asset].size:.4f}), SL: {sl_price:.2f}, "
                f"Trail: {self.config.trailing_stop_pct}%@{self.config.trailing_activation_pct}%"
            )

        except Exception as e:
            error_str = str(e).lower()
            logger.error(f"{asset}: Failed to enter short: {e}")

            # Check if this is a DEFINITIVE failure (order definitely did not go through)
            # vs an UNCERTAIN failure (order might have gone through)
            definitive_failures = [
                'insufficient margin',
                'insufficient balance',
                'not enough margin',
                'margin requirement',
                'invalid size',
                'invalid amount',
                'min order',
                'minimum order',
                'max position',
                'position limit',
            ]

            is_definitive_failure = any(fail in error_str for fail in definitive_failures)

            if is_definitive_failure:
                # Order definitely failed - safe to clear and allow retry later
                logger.warning(f"{asset}: Definitive failure - clearing position lock")
                self.active_trades[asset] = None
            else:
                # Uncertain failure (timeout, network, etc.) - keep locked to be safe
                logger.error(f"{asset}: KEEPING POSITION LOCKED - check exchange manually and restart bot if no position exists")

    async def exit_short(self, asset: str, current_price: float, reason: str):
        """Exit a short position."""
        trade = self.active_trades[asset]
        if trade is None:
            return

        # Check if halted
        if self.halted.get(asset, False):
            logger.warning(f"{asset}: Trading halted - skipping exit")
            return

        client = self.clients[asset]

        try:
            from exchange.base import OrderSide, OrderType

            # CRITICAL: Always get actual position from exchange before exit
            # This prevents sending wrong size and creating opposite positions
            positions = await client.get_positions()
            actual_short = None
            actual_long = None

            for pos in positions:
                if pos.side == "SHORT" and pos.size > 0.0001:
                    actual_short = pos
                elif pos.side == "LONG" and pos.size > 0.0001:
                    actual_long = pos

            # Check for position mismatch - found LONG instead of SHORT
            if actual_long is not None:
                logger.error(f"{asset}: POSITION MISMATCH - Expected SHORT but found LONG {actual_long.size:.6f}!")
                logger.error(f"{asset}: Clearing tracked trade - DO NOT send BUY order")
                self.active_trades[asset] = None
                return  # Do NOT send a BUY order - would make LONG bigger

            # No SHORT on exchange - nothing to close
            if actual_short is None:
                logger.warning(f"{asset}: No SHORT position on exchange - already closed?")
                self.active_trades[asset] = None
                return  # Nothing to close

            # Use ACTUAL exchange size, not tracked size
            size_to_close = actual_short.size

            if abs(size_to_close - trade.size) > 0.0001:
                logger.warning(f"{asset}: Size mismatch - tracked: {trade.size:.6f}, exchange: {size_to_close:.6f}")
                logger.info(f"{asset}: Using exchange size {size_to_close:.6f} for exit")

            # Use place_order_until_filled for critical exit orders
            # Pass closing_side so verification knows what to look for
            if hasattr(client, 'place_order_until_filled'):
                order = await client.place_order_until_filled(
                    side=OrderSide.BUY,
                    size=size_to_close,  # Use EXCHANGE size
                    max_attempts=5,
                    closing_side="SHORT",  # Tell verification we're closing a SHORT
                )
            else:
                # Fallback to regular place_order for clients without the method
                order = await client.place_order(
                    side=OrderSide.BUY,
                    size=size_to_close,  # Use EXCHANGE size
                    order_type=OrderType.MARKET,
                )

            fill_price = order.avg_fill_price or current_price

            pnl_pct = (trade.entry_price - fill_price) / trade.entry_price * 100
            notional = self.config.margin_per_trade * self.config.leverage
            pnl_usd = pnl_pct / 100 * notional

            # Update stats
            self.stats[asset]["trades"] += 1
            self.stats[asset]["pnl"] += pnl_usd
            if pnl_pct > 0:
                self.stats[asset]["wins"] += 1
            else:
                self.stats[asset]["losses"] += 1

            if reason == "trailing_stop":
                self.stats[asset]["trailing_exits"] += 1
            elif reason == "lower_bb":
                self.stats[asset]["lower_bb_exits"] += 1
            elif reason == "stop_loss":
                self.stats[asset]["sl_exits"] += 1

            # Log trade
            entry_rsi = getattr(trade, '_entry_rsi', 0)
            self._log_trade(asset, trade, fill_price, pnl_pct, pnl_usd, reason, entry_rsi)

            win_rate = self.stats[asset]["wins"] / self.stats[asset]["trades"] * 100

            max_profit_pct = (trade.entry_price - trade.best_price) / trade.entry_price * 100

            logger.info(
                f"{asset}: EXITED SHORT @ {fill_price:.2f} ({reason}), "
                f"PnL: {pnl_pct:+.2f}% (${pnl_usd:+.2f}), MaxProfit: {max_profit_pct:.2f}%"
            )
            logger.info(
                f"{asset}: Stats - Trades: {self.stats[asset]['trades']}, WR: {win_rate:.1f}%, "
                f"Trail: {self.stats[asset]['trailing_exits']}, BB: {self.stats[asset]['lower_bb_exits']}, "
                f"SL: {self.stats[asset]['sl_exits']}, PnL: ${self.stats[asset]['pnl']:.2f}"
            )

            # CRITICAL: Verify position is actually closed
            await asyncio.sleep(0.5)  # Brief delay for position to update
            try:
                positions = await client.get_positions()
                remaining_short = None
                for pos in positions:
                    if pos.side == "SHORT" and pos.size > 0:
                        remaining_short = pos.size
                        break

                if remaining_short is not None and remaining_short > 0.001:
                    logger.error(
                        f"{asset}: EXIT INCOMPLETE - Still have {remaining_short:.4f} SHORT position!"
                    )
                    logger.error(f"{asset}: Manual intervention may be required")
                    # Keep the trade active so bot doesn't re-enter
                    self.active_trades[asset].size = remaining_short
                    return  # Don't clear active_trades
                else:
                    logger.info(f"{asset}: Exit verified - position fully closed")
            except Exception as ve:
                logger.warning(f"{asset}: Could not verify exit: {ve}")

            self.active_trades[asset] = None

        except Exception as e:
            logger.error(f"{asset}: Failed to exit short: {e}")
            # CRITICAL: Check if position still exists after failed exit
            try:
                positions = await client.get_positions()
                for pos in positions:
                    if pos.side == "SHORT" and pos.size > 0:
                        logger.error(f"{asset}: Position still open after failed exit: {pos.size:.4f}")
                        # Update tracked size in case it changed
                        self.active_trades[asset].size = pos.size
                        return  # Keep position tracked
            except:
                pass

    async def update(self, asset: str, current_price: float):
        """Update strategy with new price. ALWAYS uses real exchange positions."""
        # Check if asset is halted
        if self.halted.get(asset, False):
            return  # Skip all processing for halted assets

        # Only add price to history every 10 seconds (for consistent BB calculation)
        current_time = time.time()
        if current_time - self._last_price_time[asset] >= self._price_collection_interval:
            self.price_history[asset].append(current_price)
            self._last_price_time[asset] = current_time

        # ALWAYS check real exchange position - this is the source of truth
        client = self.clients[asset]
        try:
            positions = await client.get_positions()
            real_short = None
            real_long = None

            for pos in positions:
                if pos.size > 0.0001:
                    if pos.side == "SHORT":
                        real_short = pos
                    elif pos.side == "LONG":
                        real_long = pos

            # If we have a LONG in this SHORT-only strategy, close it
            if real_long is not None:
                logger.warning(f"{asset}: Found unexpected LONG {real_long.size:.6f} - closing")
                await self._close_position(asset, real_long)
                return

            # If we have a SHORT, manage it
            if real_short is not None:
                # Adopt into active_trades if not already tracked
                if self.active_trades[asset] is None:
                    logger.info(f"{asset}: Adopting real SHORT {real_short.size:.6f} into tracking")
                    sl_price = real_short.entry_price * (1 + self.config.stop_loss_pct / 100) if real_short.entry_price > 0 else current_price * (1 + self.config.stop_loss_pct / 100)
                    self.active_trades[asset] = RSIBBTrade(
                        asset=asset,
                        entry_time=datetime.now(),
                        entry_price=real_short.entry_price if real_short.entry_price > 0 else current_price,
                        size=real_short.size,
                        stop_loss_price=sl_price,
                        best_price=current_price,
                        trailing_active=False,
                    )
                    self.active_trades[asset]._entry_rsi = 0

                # Update tracked size to match reality
                if self.active_trades[asset] is not None:
                    self.active_trades[asset].size = real_short.size

                # Check position sanity
                if not await self._check_position_sanity(asset, current_price):
                    return

                # Check exit conditions
                exit_reason = await self.check_exit(asset, current_price)
                if exit_reason:
                    await self.exit_short(asset, current_price, exit_reason)
                return

            # No position - clear tracking if stale and check for entry
            if self.active_trades[asset] is not None:
                logger.info(f"{asset}: No real position but had tracked - clearing stale tracking")
                self.active_trades[asset] = None

            # Check entry
            entry_rsi = await self.check_entry(asset, current_price)
            if entry_rsi is not None:
                await self.enter_short(asset, current_price, entry_rsi)

        except Exception as e:
            logger.error(f"{asset}: Error checking positions: {e}")

    async def _close_position(self, asset: str, pos):
        """Close a position (used for closing unexpected LONGs)."""
        client = self.clients[asset]
        try:
            from exchange.base import OrderSide, OrderType
            close_side = OrderSide.SELL if pos.side == "LONG" else OrderSide.BUY
            await client.place_order(
                side=close_side,
                size=pos.size,
                order_type=OrderType.MARKET,
                slippage_pct=0.5,
            )
            logger.info(f"{asset}: Closed {pos.side} position of {pos.size:.6f}")
        except Exception as e:
            logger.error(f"{asset}: Failed to close position: {e}")

    async def run_asset(self, asset: str):
        """Run strategy loop for a single asset."""
        client = self.clients[asset]

        logger.info(f"{asset}: Starting RSI+BB strategy (RSI entry > {self.rsi_entry.get(asset, 70)}) - using shared JSON prices")
        log_counter = 0
        first_bbo = True
        dust_check_counter = 0
        consecutive_network_errors = 0
        consecutive_stale_prices = 0

        while self.running:
            # Check if halted
            if self.halted.get(asset, False):
                logger.debug(f"{asset}: Trading halted - sleeping")
                await asyncio.sleep(30)  # Check less frequently when halted
                continue
            try:
                # Read price from shared JSON instead of API call
                if not is_price_fresh(asset, max_age=PRICE_MAX_AGE):
                    consecutive_stale_prices += 1
                    if consecutive_stale_prices >= 10:
                        logger.warning(f"{asset}: Prices stale for too long, is price_fetcher.py running?")
                        consecutive_stale_prices = 0
                    await asyncio.sleep(2)
                    continue

                consecutive_stale_prices = 0
                bbo_data = get_bbo(asset)
                mid_price = bbo_data.get("mid", 0)
                bid_price = bbo_data.get("bid", 0)
                ask_price = bbo_data.get("ask", 0)

                if mid_price <= 0:
                    logger.warning(f"{asset}: Invalid price from JSON: {mid_price}")
                    await asyncio.sleep(2)
                    continue

                # Reset network error counter on successful operation
                if consecutive_network_errors > 0:
                    logger.info(f"{asset}: Network connection restored after {consecutive_network_errors} errors")
                    consecutive_network_errors = 0

                if first_bbo:
                    logger.info(f"{asset}: First price from JSON - Bid: {bid_price:.2f}, Ask: {ask_price:.2f}")
                    first_bbo = False

                trade = self.active_trades[asset]
                if trade:
                    pass  # Position logging handled below
                else:
                    # Log status periodically when not in a trade
                    log_counter += 1
                    if log_counter >= 4:  # Log every ~2 minutes (4 * 30s)
                        prices = list(self.price_history[asset])
                        price_count = len(prices)
                        if price_count >= max(self.config.bb_period, self.config.rsi_period + 1):
                            _, _, upper_bb = self.calculate_bb(prices)
                            rsi = self.calculate_rsi(prices)
                            rsi_thresh = self.rsi_entry.get(asset, 70)
                            logger.info(
                                f"{asset}: Watching - Price: {mid_price:.2f}, Upper BB: {upper_bb:.2f}, "
                                f"RSI: {rsi:.1f}/{rsi_thresh} (need price > BB AND RSI > {rsi_thresh})"
                            )
                        else:
                            logger.info(f"{asset}: Building history - {price_count}/{self.config.bb_period} prices collected")
                        log_counter = 0

                if trade:
                    log_counter += 1
                    if log_counter >= 12:  # Log every ~60s
                        pnl_pct = (trade.entry_price - bid_price) / trade.entry_price * 100
                        pnl_usd = pnl_pct / 100 * self.config.margin_per_trade * self.config.leverage
                        trail_status = "ACTIVE" if trade.trailing_active else "inactive"
                        trail_price = trade.best_price * (1 + self.config.trailing_stop_pct / 100) if trade.trailing_active else 0

                        prices = list(self.price_history[asset])
                        lower_bb, _, _ = self.calculate_bb(prices) if len(prices) >= self.config.bb_period else (None, None, None)
                        lower_str = f"{lower_bb:.2f}" if lower_bb else "N/A"

                        logger.info(
                            f"{asset}: Bid: {bid_price:.2f}, Best: {trade.best_price:.2f}, "
                            f"Trail({trail_status}): {trail_price:.2f}, LowerBB: {lower_str}, "
                            f"SL: {trade.stop_loss_price:.2f}, PnL: ${pnl_usd:+.2f}"
                        )
                        log_counter = 0
                # Note: log_counter is already incremented and reset in the status logging blocks above

                # Use bid for exit checks when in trade
                await self.update(asset, bid_price if trade else mid_price)

                # Periodic dust cleanup when not in trade (every ~5 minutes)
                if not self.active_trades[asset]:
                    dust_check_counter += 1
                    if dust_check_counter >= 30:  # 30 * 10s = 5 minutes
                        await self._cleanup_dust_positions(asset, mid_price)
                        dust_check_counter = 0

                # Fast checks when in trade, slower for entries
                if self.active_trades[asset]:
                    await asyncio.sleep(3)  # Reduced from 1s to avoid rate limits
                else:
                    await asyncio.sleep(10)

            except NetworkError as e:
                consecutive_network_errors += 1

                if consecutive_network_errors >= MAX_CONSECUTIVE_NETWORK_ERRORS:
                    logger.error(f"{asset}: CRITICAL - {consecutive_network_errors} consecutive network errors, halting trading")
                    self.halted[asset] = True
                    continue

                # Calculate backoff with exponential increase, capped at max
                backoff = min(NETWORK_ERROR_BACKOFF_BASE * (2 ** (consecutive_network_errors - 1)), NETWORK_ERROR_BACKOFF_MAX)
                logger.warning(f"{asset}: Network error ({consecutive_network_errors}/{MAX_CONSECUTIVE_NETWORK_ERRORS}), retrying in {backoff:.1f}s: {e}")

                # Attempt reconnection if many errors
                if consecutive_network_errors >= 3 and not client.paper_mode:
                    logger.info(f"{asset}: Attempting reconnection...")
                    try:
                        await client.disconnect()
                        await asyncio.sleep(1)
                        if await client.connect():
                            logger.info(f"{asset}: Reconnected successfully")
                        else:
                            logger.warning(f"{asset}: Reconnection failed")
                    except Exception as reconnect_err:
                        logger.warning(f"{asset}: Reconnection attempt failed: {reconnect_err}")

                await asyncio.sleep(backoff)

            except Exception as e:
                logger.error(f"{asset}: Error in strategy loop: {e}")
                await asyncio.sleep(5)

    async def _bootstrap_price_history(self):
        """
        Bootstrap price history from Binance candles so bot can trade immediately.
        Uses Binance public API (no auth needed) for reliable candle data.
        """
        import time as time_module
        import aiohttp

        logger.info("Bootstrapping price history from Binance...")

        # Map our assets to Binance symbols
        binance_symbols = {
            "BTC": "BTCUSDT",
            "ETH": "ETHUSDT",
            "SOL": "SOLUSDT",
        }

        async with aiohttp.ClientSession() as session:
            for asset, client in self.clients.items():
                try:
                    symbol = binance_symbols.get(asset)
                    if not symbol:
                        logger.warning(f"{asset}: No Binance symbol mapping")
                        continue

                    # Fetch 50 1-minute candles from Binance (need 40+ for BB period)
                    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1m&limit=50"
                    async with session.get(url) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            # Binance kline format: [open_time, open, high, low, close, volume, ...]
                            # Only use close prices to avoid artificial OHLC swings in RSI
                            for candle in data:
                                self.price_history[asset].append(float(candle[4]))  # close only

                            self._last_price_time[asset] = time_module.time()
                            logger.info(f"{asset}: Bootstrapped {len(self.price_history[asset])} close prices from {len(data)} Binance candles")
                        else:
                            logger.warning(f"{asset}: Binance API returned {resp.status}")

                except Exception as e:
                    logger.warning(f"{asset}: Failed to bootstrap from Binance: {e}")

    async def run(self):
        """Run strategy for all assets."""
        self.running = True

        for asset, client in self.clients.items():
            if not client.connected:
                await client.connect()

        # CRITICAL: Check for existing positions on startup
        # This prevents entering duplicate positions if bot restarts with open positions
        await self._check_existing_positions()

        # Bootstrap price history from candles for immediate trading
        await self._bootstrap_price_history()

        logger.info("=" * 70)
        logger.info("RSI+BB STRATEGY - Starting")
        logger.info("=" * 70)
        logger.info(f"Assets: {list(self.clients.keys())}")
        logger.info(f"Entry: Price > Upper BB({self.config.bb_period}) AND RSI({self.config.rsi_period}) > threshold")
        logger.info(f"RSI thresholds: {self.rsi_entry}")
        logger.info(f"Exit: Trail {self.config.trailing_stop_pct}%@{self.config.trailing_activation_pct}% OR Lower BB OR SL {self.config.stop_loss_pct}%")
        logger.info(f"Position: ${self.config.margin_per_trade} x {self.config.leverage}x = ${self.config.margin_per_trade * self.config.leverage}")
        logger.info("=" * 70)

        tasks = [self.run_asset(asset) for asset in self.clients.keys()]
        await asyncio.gather(*tasks)

    async def _cleanup_dust_positions(self, asset: str, current_price: float):
        """Clean up any tiny residual positions (dust) that shouldn't exist."""
        if self.active_trades[asset] is not None:
            return  # Don't interfere with active trades

        if self.halted.get(asset, False):
            return  # Don't clean up if halted

        try:
            client = self.clients[asset]
            positions = await client.get_positions()

            for pos in positions:
                notional = pos.size * current_price
                # Clean up positions worth less than $50
                if notional < 50 and notional > 0:
                    logger.info(f"{asset}: Cleaning up dust position: {pos.size:.6f} {pos.side} (${notional:.2f})")
                    from exchange.base import OrderSide, OrderType
                    close_side = OrderSide.SELL if pos.side == "LONG" else OrderSide.BUY
                    try:
                        await client.place_order(
                            side=close_side,
                            size=pos.size,
                            order_type=OrderType.MARKET,
                        )
                        logger.info(f"{asset}: Dust position closed")
                    except Exception as e:
                        logger.warning(f"{asset}: Failed to close dust: {e}")

        except Exception as e:
            logger.warning(f"{asset}: Error checking for dust: {e}")

    async def _check_existing_positions(self):
        """Check for existing positions on exchange and populate active_trades."""
        logger.info("Checking for existing positions on exchange...")

        for asset, client in self.clients.items():
            try:
                positions = await client.get_positions()
                for pos in positions:
                    if pos.side == "SHORT" and pos.size > 0:
                        # Found existing short position - create active trade entry
                        logger.warning(
                            f"{asset}: FOUND EXISTING SHORT POSITION - Size: {pos.size:.4f}, "
                            f"Entry: {pos.entry_price:.2f}, PnL: ${pos.unrealized_pnl:.2f}"
                        )

                        # Create a trade entry so bot won't enter again
                        sl_price = pos.entry_price * (1 + self.config.stop_loss_pct / 100)
                        self.active_trades[asset] = RSIBBTrade(
                            asset=asset,
                            entry_time=datetime.now(),  # Unknown, use now
                            entry_price=pos.entry_price,
                            size=pos.size,
                            stop_loss_price=sl_price,
                            best_price=pos.entry_price,  # Unknown, use entry
                            trailing_active=False,
                        )
                        self.active_trades[asset]._entry_rsi = 0  # Unknown

                        logger.warning(f"{asset}: Will manage existing position (SL: {sl_price:.2f})")

            except Exception as e:
                logger.error(f"{asset}: Error checking positions: {e}")

        # Summary
        active_count = sum(1 for t in self.active_trades.values() if t is not None)
        if active_count > 0:
            logger.warning(f"Found {active_count} existing position(s) - bot will manage them")
        else:
            logger.info("No existing positions found - starting fresh")

    async def stop(self):
        """Stop strategy."""
        self.running = False

        logger.info("=" * 70)
        logger.info("FINAL STATS")
        logger.info("=" * 70)
        total_trades = 0
        total_wins = 0
        total_pnl = 0

        for asset, stats in self.stats.items():
            if stats["trades"] > 0:
                wr = stats["wins"] / stats["trades"] * 100
                logger.info(
                    f"{asset}: {stats['trades']} trades, WR: {wr:.1f}%, "
                    f"Trail: {stats['trailing_exits']}, BB: {stats['lower_bb_exits']}, "
                    f"SL: {stats['sl_exits']}, PnL: ${stats['pnl']:.2f}"
                )
                total_trades += stats["trades"]
                total_wins += stats["wins"]
                total_pnl += stats["pnl"]

        if total_trades > 0:
            logger.info(f"TOTAL: {total_trades} trades, WR: {total_wins/total_trades*100:.1f}%, PnL: ${total_pnl:.2f}")
