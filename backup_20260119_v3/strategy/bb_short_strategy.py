"""
Bollinger Band Short Strategy (Mean Reversion)

Entry: SHORT when price closes above upper Bollinger Band
Exit: Pure SL/TP (no time-based exits)

Verified profitable on BTC, ETH, SOL with:
- SL: 0.3%, TP: 0.4%
- Win rate ~46% vs 42.9% breakeven
- Positive EV: +0.02-0.03% per trade
"""

import asyncio
import logging
import csv
import os
from datetime import datetime
from typing import Optional, Dict
from dataclasses import dataclass
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)

# Trade history CSV path
TRADE_HISTORY_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(TRADE_HISTORY_DIR, exist_ok=True)


@dataclass
class BBShortTrade:
    """Active short trade."""
    asset: str
    entry_time: datetime
    entry_price: float
    size: float
    stop_loss_price: float
    take_profit_price: float


@dataclass
class BBShortConfig:
    """Configuration for BB Short strategy."""
    bb_period: int = 20
    bb_std: float = 2.0
    stop_loss_pct: float = 0.30
    take_profit_pct: float = 0.40
    margin_per_trade: float = 100.0  # $100 margin per asset
    leverage: float = 20.0  # 20x leverage = $2000 notional per trade


class BBShortStrategy:
    """
    Bollinger Band Short Mean Reversion Strategy.

    Shorts when price is overbought (above upper BB).
    Pure SL/TP exits - no time-based exits.
    """

    def __init__(
        self,
        exchange_clients: Dict[str, any],  # {"BTC": client, "ETH": client, "SOL": client}
        config: BBShortConfig = None,
    ):
        self.clients = exchange_clients
        self.config = config or BBShortConfig()

        # Price history for BB calculation (need bb_period + buffer)
        self.price_history: Dict[str, deque] = {
            asset: deque(maxlen=self.config.bb_period + 10)
            for asset in exchange_clients.keys()
        }

        # Active trades per asset
        self.active_trades: Dict[str, Optional[BBShortTrade]] = {
            asset: None for asset in exchange_clients.keys()
        }

        # Stats
        self.stats = {
            asset: {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0}
            for asset in exchange_clients.keys()
        }

        self.running = False

        # Trade history CSV file
        self.trade_history_file = os.path.join(TRADE_HISTORY_DIR, "bb_short_trades.csv")
        self._init_trade_history()

    def _init_trade_history(self):
        """Initialize trade history CSV if it doesn't exist."""
        if not os.path.exists(self.trade_history_file):
            with open(self.trade_history_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'asset', 'entry_time', 'exit_time', 'entry_price',
                    'exit_price', 'size', 'pnl_pct', 'pnl_usd', 'exit_reason',
                    'sl_price', 'tp_price', 'hold_seconds'
                ])

    def _log_trade(self, asset: str, trade, exit_price: float, pnl_pct: float, pnl_usd: float, exit_reason: str):
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
                    trade.size,
                    round(pnl_pct, 4),
                    round(pnl_usd, 2),
                    exit_reason,
                    trade.stop_loss_price,
                    trade.take_profit_price,
                    round(hold_seconds, 1)
                ])
        except Exception as e:
            logger.error(f"Failed to log trade: {e}")

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

    async def check_entry(self, asset: str, current_price: float) -> bool:
        """Check if we should enter a short position."""
        if self.active_trades[asset] is not None:
            return False  # Already in trade

        prices = list(self.price_history[asset])
        if len(prices) < self.config.bb_period:
            return False

        _, _, upper_bb = self.calculate_bb(prices)

        if upper_bb and current_price > upper_bb:
            logger.info(f"{asset}: Price {current_price:.2f} > Upper BB {upper_bb:.2f} - SHORT SIGNAL")
            return True

        return False

    async def check_exit(self, asset: str, current_price: float) -> Optional[str]:
        """Check if we should exit position. Returns exit reason or None."""
        trade = self.active_trades[asset]
        if trade is None:
            return None

        # For short: profit when price goes down, loss when price goes up
        pnl_pct = (trade.entry_price - current_price) / trade.entry_price * 100

        if current_price >= trade.stop_loss_price:
            return "stop_loss"
        elif current_price <= trade.take_profit_price:
            return "take_profit"

        return None

    async def enter_short(self, asset: str, current_price: float):
        """Enter a short position."""
        client = self.clients[asset]

        # Calculate position size: margin × leverage = notional
        notional = self.config.margin_per_trade * self.config.leverage
        size = notional / current_price

        # Calculate SL/TP prices (for short: SL above entry, TP below entry)
        sl_price = current_price * (1 + self.config.stop_loss_pct / 100)
        tp_price = current_price * (1 - self.config.take_profit_pct / 100)

        try:
            from exchange.base import OrderSide, OrderType

            order = await client.place_order(
                side=OrderSide.SELL,
                size=size,
                order_type=OrderType.MARKET,
            )

            fill_price = order.avg_fill_price or current_price

            self.active_trades[asset] = BBShortTrade(
                asset=asset,
                entry_time=datetime.now(),
                entry_price=fill_price,
                size=size,
                stop_loss_price=sl_price,
                take_profit_price=tp_price,
            )

            logger.info(f"{asset}: ENTERED SHORT @ {fill_price:.2f}, Size: ${notional:.0f} ({size:.4f}), SL: {sl_price:.2f}, TP: {tp_price:.2f}")

        except Exception as e:
            logger.error(f"{asset}: Failed to enter short: {e}")

    async def exit_short(self, asset: str, current_price: float, reason: str):
        """Exit a short position."""
        trade = self.active_trades[asset]
        if trade is None:
            return

        client = self.clients[asset]

        try:
            from exchange.base import OrderSide, OrderType

            order = await client.place_order(
                side=OrderSide.BUY,  # Buy to close short
                size=trade.size,
                order_type=OrderType.MARKET,
            )

            fill_price = order.avg_fill_price or current_price

            # Calculate PnL (for short: profit = entry - exit)
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

            # Log trade to CSV
            self._log_trade(asset, trade, fill_price, pnl_pct, pnl_usd, reason)

            win_rate = self.stats[asset]["wins"] / self.stats[asset]["trades"] * 100 if self.stats[asset]["trades"] > 0 else 0

            logger.info(f"{asset}: EXITED SHORT @ {fill_price:.2f} ({reason}), PnL: {pnl_pct:+.2f}% (${pnl_usd:+.2f})")
            logger.info(f"{asset}: Stats - Trades: {self.stats[asset]['trades']}, WR: {win_rate:.1f}%, Total PnL: ${self.stats[asset]['pnl']:.2f}")

            self.active_trades[asset] = None

        except Exception as e:
            logger.error(f"{asset}: Failed to exit short: {e}")

    async def update(self, asset: str, current_price: float):
        """Update strategy with new price."""
        # Add to price history
        self.price_history[asset].append(current_price)

        # Check for exit first
        exit_reason = await self.check_exit(asset, current_price)
        if exit_reason:
            await self.exit_short(asset, current_price, exit_reason)
            return

        # Check for entry
        if await self.check_entry(asset, current_price):
            await self.enter_short(asset, current_price)

    async def run_asset(self, asset: str):
        """Run strategy loop for a single asset."""
        client = self.clients[asset]

        logger.info(f"{asset}: Starting BB Short strategy")
        log_counter = 0  # Only log every 12th check (once per minute) when in trade

        while self.running:
            try:
                bbo = await client.get_bbo()
                # For shorts: use bid price for exit checks (that's what you get when buying to close)
                # Use mid price for entry signals
                mid_price = (bbo.bid_price + bbo.ask_price) / 2

                # Log price if we have an active trade (every 60s to avoid spam)
                trade = self.active_trades[asset]
                if trade:
                    log_counter += 1
                    if log_counter >= 12:  # Log every ~60 seconds (12 * 5s)
                        pnl_pct = (trade.entry_price - bbo.bid_price) / trade.entry_price * 100
                        pnl_usd = pnl_pct / 100 * self.config.margin_per_trade * self.config.leverage
                        logger.info(f"{asset}: Bid: {bbo.bid_price:.2f}, TP: {trade.take_profit_price:.2f}, SL: {trade.stop_loss_price:.2f}, PnL: ${pnl_usd:+.2f}")
                        log_counter = 0
                else:
                    log_counter = 0

                await self.update(asset, bbo.bid_price if self.active_trades[asset] else mid_price)

                # When in a trade: check every 5 seconds for faster SL/TP response
                # When not in trade: check every 30 seconds for entry signals
                if self.active_trades[asset]:
                    await asyncio.sleep(5)
                else:
                    await asyncio.sleep(30)

            except Exception as e:
                logger.error(f"{asset}: Error in strategy loop: {e}")
                await asyncio.sleep(5)

    async def run(self):
        """Run strategy for all assets concurrently."""
        self.running = True

        # Connect all clients
        for asset, client in self.clients.items():
            if not client.connected:
                await client.connect()

        logger.info("="*60)
        logger.info("BB SHORT STRATEGY - Starting")
        logger.info(f"Assets: {list(self.clients.keys())}")
        logger.info(f"Config: BB({self.config.bb_period}), SL: {self.config.stop_loss_pct}%, TP: {self.config.take_profit_pct}%")
        logger.info("="*60)

        # Run all assets concurrently
        tasks = [self.run_asset(asset) for asset in self.clients.keys()]
        await asyncio.gather(*tasks)

    async def stop(self):
        """Stop the strategy and close all positions."""
        self.running = False

        # Close any open positions
        for asset, trade in self.active_trades.items():
            if trade is not None:
                try:
                    bbo = await self.clients[asset].get_bbo()
                    mid_price = (bbo.bid_price + bbo.ask_price) / 2
                    await self.exit_short(asset, mid_price, "shutdown")
                except Exception as e:
                    logger.error(f"{asset}: Failed to close position on shutdown: {e}")

        # Print final stats
        logger.info("="*60)
        logger.info("FINAL STATS")
        logger.info("="*60)
        total_trades = 0
        total_wins = 0
        total_pnl = 0
        for asset, stats in self.stats.items():
            if stats["trades"] > 0:
                wr = stats["wins"] / stats["trades"] * 100
                logger.info(f"{asset}: {stats['trades']} trades, WR: {wr:.1f}%, PnL: ${stats['pnl']:.2f}")
                total_trades += stats["trades"]
                total_wins += stats["wins"]
                total_pnl += stats["pnl"]

        if total_trades > 0:
            logger.info(f"TOTAL: {total_trades} trades, WR: {total_wins/total_trades*100:.1f}%, PnL: ${total_pnl:.2f}")
